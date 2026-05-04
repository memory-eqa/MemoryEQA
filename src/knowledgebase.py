"""
Structured Multi-modal Hierarchical Memory for Memory-Centric EQA.

Implements the four key innovations from the paper:
  1. Structured multi-modal hierarchical memory (global + local)
  2. Viewpoint-aware update mechanism
  3. Entropy-based adaptive retrieval
  4. Global memory queries for planner integration

Replaces the previous flat DynamicKnowledgeBase.
"""

import faiss
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch


@dataclass
class LocalMemoryEntry:
    """A single per-step observation stored in local memory (paper Sec 3.2, L_i).

    L_i = {I_i, o_i, p_i, r_i, m_i, f_i}
    """
    entry_id: int                          # I_i: unique memory index
    image: Image.Image                     # o_i: observation image
    position: np.ndarray                   # p_i: 3D position (habitat coords)
    rotation: np.ndarray                   # r_i: quaternion [w, x, y, z]
    structured_text: str                   # m_i: structured text (room, objects, captions)
    feature_vector: np.ndarray             # f_i: combined image+text feature vector
    image_vector: np.ndarray               # image-only feature vector
    text_vector: np.ndarray                # text-only feature vector
    room_type: str                         # room classification
    objects: List[Dict]                    # detected objects [{cls, caption, pos}, ...]
    step: int                              # step index when recorded
    timestamp: float = field(default_factory=time.time)


class StructuredMemory:
    """Structured multi-modal hierarchical memory.

    Local memory: per-step observations with structured entries.
    Global memory: aggregated scene-level room/object information.

    Replaces DynamicKnowledgeBase with:
      - Viewpoint-aware update (Innovation 2)
      - Entropy-based adaptive retrieval (Innovation 3)
      - Global memory queries for planner integration (Innovation 4)
    """

    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device

        # --- Embedding model loading (preserved from DynamicKnowledgeBase) ---
        self.text_embedder = None
        self.image_model = None
        self.preprocess = None

        text_model_name = cfg.text.split("/")[-1]
        visual_model_name = cfg.visual.split("/")[-1]

        if text_model_name == visual_model_name and visual_model_name in ["clip-vit-large-patch14"]:
            from transformers import CLIPProcessor, CLIPModel
            self.text_embedder = CLIPModel.from_pretrained(cfg.text).to(device)
            self.image_model = self.text_embedder
            self.preprocess = CLIPProcessor.from_pretrained(cfg.text)
        elif text_model_name == visual_model_name and visual_model_name in ["blip2-opt-2.7b"]:
            from transformers import Blip2Processor, Blip2Model
            self.text_embedder = Blip2Model.from_pretrained(
                cfg.text, torch_dtype=torch.float16
            ).to(device)
            self.image_model = self.text_embedder
            self.preprocess = Blip2Processor.from_pretrained(cfg.text)
        else:
            from sentence_transformers import SentenceTransformer
            import clip
            self.text_embedder = SentenceTransformer(cfg.text, device=device)
            self.image_model, self.preprocess = clip.load(cfg.visual, device=device)

        # --- FAISS index for feature vectors ---
        self.index = faiss.IndexFlatL2(cfg.dim)

        # --- Local memory store ---
        self.local_entries: List[LocalMemoryEntry] = []
        self.next_entry_id: int = 0

        # --- Global memory: language-enhanced scene map ---
        self.global_rooms: Dict[str, Dict] = {}

        # --- Config parameters with defaults ---
        self.use_viewpoint_update = cfg.get("use_viewpoint_update", True)
        self.viewpoint_beta_p = cfg.get("viewpoint_beta_p", 0.5)
        self.viewpoint_beta_r = cfg.get("viewpoint_beta_r", 0.5)
        self.viewpoint_visual_sim = cfg.get("viewpoint_visual_sim", 0.85)
        self.viewpoint_obj_sim = cfg.get("viewpoint_obj_sim", 0.7)
        self.use_entropy_retrieval = cfg.get("use_entropy_retrieval", True)
        self.entropy_alpha_e = cfg.get("entropy_alpha_e", 0.3)
        self.entropy_alpha_s = cfg.get("entropy_alpha_s", 0.7)
        self.entropy_search_multiplier = cfg.get("entropy_search_multiplier", 3)
        self.max_retrieval_num = cfg.get("max_retrieval_num", 5)
        self.lambda_sim = cfg.get("lambda_sim", 0.5)

        logging.info(
            f"StructuredMemory initialized: viewpoint_update={self.use_viewpoint_update}, "
            f"entropy_retrieval={self.use_entropy_retrieval}"
        )

    # ========================================================================
    #  Encoding (preserved pattern from DynamicKnowledgeBase)
    # ========================================================================

    def _encode_entry(
        self, text: str, image: Image.Image, device: str = "cuda"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode text and image into feature vectors.

        Returns (combined_vector, image_vector, text_vector).
        """
        image_vector = None
        text_vector = None

        if text:
            text_inputs = self.preprocess(
                text=text, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            text_vector = (
                self.text_embedder.get_text_features(**text_inputs)
                .cpu().detach().numpy()
            )
        if image:
            image_inputs = self.preprocess(
                images=image, return_tensors="pt", padding=True, truncation=True
            ).to(self.image_model.device if hasattr(self.image_model, 'device') else device)
            image_vector = (
                self.image_model.get_image_features(**image_inputs)
                .cpu().detach().numpy()
            )

        if image_vector is not None and text_vector is not None:
            combined_vector = np.concatenate([image_vector, text_vector], axis=1)
        elif image_vector is not None:
            combined_vector = image_vector
        elif text_vector is not None:
            combined_vector = text_vector
        else:
            raise ValueError("Both text and image are None in _encode_entry")

        return combined_vector, image_vector, text_vector

    # ========================================================================
    #  Structured text construction
    # ========================================================================

    def _build_structured_text(
        self, room_type: str, position: np.ndarray,
        objects: List[Dict], caption: str
    ) -> str:
        """Build structured text m_i matching paper's format.

        Format: "Room: <room>. Objects: [<cls>: <caption> at <pos> | ...]. Caption: <caption>"
        """
        obj_strs = []
        for obj in objects:
            pos_str = f"({obj['pos'][0]:.2f}, {obj['pos'][1]:.2f}, {obj['pos'][2]:.2f})"
            obj_strs.append(f"{obj['cls']}: {obj['caption']} at {pos_str}")
        objs_text = " | ".join(obj_strs) if obj_strs else "none"
        return f"Room: {room_type}. Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}). Objects: [{objs_text}]. Caption: {caption}"

    # ========================================================================
    #  Innovation 2: Viewpoint-aware Update Mechanism
    # ========================================================================

    def _compute_viewpoint_similarity(
        self,
        pos_new: np.ndarray,
        rot_new: np.ndarray,
        pos_existing: np.ndarray,
        rot_existing: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute position and rotation similarity between two viewpoints.

        Paper Sec 3.2: thresholds beta_p and beta_r determine whether
        two viewpoints observe the same region.

        Returns (pos_sim, rot_sim) in [0, 1].
        """
        # Position similarity: hfov_dist = lateral field of view at self.cfg.get("max_depth", 10.0)
        max_depth = self.cfg.get("max_depth", 10.0)
        hfov = self.cfg.get("hfov", 120)
        hfov_rad = np.radians(hfov)
        hfov_dist = 2 * max_depth * np.tan(hfov_rad / 2)
        beta_p_threshold = self.viewpoint_beta_p * hfov_dist

        pos_dist = float(np.linalg.norm(np.array(pos_new) - np.array(pos_existing)))
        pos_sim = float(np.exp(-pos_dist / max(beta_p_threshold, 1e-6)))

        # Rotation similarity: geodesic distance between quaternions
        rn = np.array(rot_new).flatten()
        re = np.array(rot_existing).flatten()
        # Normalize quaternions
        rn = rn / (np.linalg.norm(rn) + 1e-8)
        re = re / (np.linalg.norm(re) + 1e-8)
        dot = float(np.abs(np.dot(rn, re)))
        dot = min(dot, 1.0)
        angle_diff = 2.0 * np.arccos(dot)
        beta_r_threshold = self.viewpoint_beta_r * max_depth
        rot_sim = float(np.exp(-angle_diff / max(beta_r_threshold / max_depth, 1e-6)))

        return pos_sim, rot_sim

    def _compute_visual_similarity(
        self, feat_new: np.ndarray, feat_existing: np.ndarray
    ) -> float:
        """Cosine similarity between feature vectors."""
        fn = feat_new.flatten()
        fe = feat_existing.flatten()
        fn_norm = fn / (np.linalg.norm(fn) + 1e-8)
        fe_norm = fe / (np.linalg.norm(fe) + 1e-8)
        return float(np.dot(fn_norm, fe_norm))

    def _compute_object_content_similarity(
        self, objs_new: List[Dict], objs_existing: List[Dict]
    ) -> float:
        """Compute object-level content similarity via Jaccard class overlap
        and positional proximity of matching objects."""
        classes_new = {obj["cls"] for obj in objs_new}
        classes_existing = {obj["cls"] for obj in objs_existing}

        if not classes_existing or not classes_new:
            return 0.0

        intersection = classes_new & classes_existing
        union = classes_new | classes_existing
        jaccard = len(intersection) / len(union) if union else 0.0

        pos_sims = []
        for cls in intersection:
            new_positions = [
                np.array(obj["pos"]) for obj in objs_new if obj["cls"] == cls
            ]
            existing_positions = [
                np.array(obj["pos"]) for obj in objs_existing if obj["cls"] == cls
            ]
            for np_pos in new_positions:
                for ep_pos in existing_positions:
                    dist = np.linalg.norm(np_pos - ep_pos)
                    pos_sims.append(float(np.exp(-dist)))

        avg_pos_sim = float(np.mean(pos_sims)) if pos_sims else 1.0
        return 0.5 * jaccard + 0.5 * avg_pos_sim

    def _should_update(
        self,
        new_entry: LocalMemoryEntry,
        existing_entry: LocalMemoryEntry,
    ) -> bool:
        """Decide whether to UPDATE existing entry or INSERT new.

        Conditions for UPDATE (paper Sec 3.2):
          1. Position close (pos_sim > 0.6)
          2. Rotation similar (rot_sim > 0.6)
          3. Visual similarity high (cosine_sim > viewpoint_visual_sim)
          4. Object content consistent (obj_sim > viewpoint_obj_sim)
        """
        pos_sim, rot_sim = self._compute_viewpoint_similarity(
            new_entry.position, new_entry.rotation,
            existing_entry.position, existing_entry.rotation
        )
        vis_sim = self._compute_visual_similarity(
            new_entry.feature_vector, existing_entry.feature_vector
        )
        obj_sim = self._compute_object_content_similarity(
            new_entry.objects, existing_entry.objects
        )

        decision = (
            pos_sim > 0.6
            and rot_sim > 0.6
            and vis_sim > self.viewpoint_visual_sim
            and obj_sim > self.viewpoint_obj_sim
        )
        logging.info(
            f"Viewpoint check: pos_sim={pos_sim:.3f} rot_sim={rot_sim:.3f} "
            f"vis_sim={vis_sim:.3f} obj_sim={obj_sim:.3f} -> {'UPDATE' if decision else 'INSERT'}"
        )
        return decision

    def _update_entry(
        self, existing: LocalMemoryEntry, new: LocalMemoryEntry, faiss_idx: int
    ):
        """Update existing entry in-place (preserve entry_id).

        Removes old FAISS vector, adds new one, updates all fields.
        """
        existing.image = new.image
        existing.position = new.position
        existing.rotation = new.rotation
        existing.structured_text = new.structured_text
        existing.feature_vector = new.feature_vector
        existing.image_vector = new.image_vector
        existing.text_vector = new.text_vector
        existing.room_type = new.room_type
        existing.objects = new.objects
        existing.step = new.step
        existing.timestamp = new.timestamp

        # Update FAISS: remove old, add new
        self.index.remove_ids(np.array([faiss_idx], dtype=np.int64))
        self.index.add(new.feature_vector)

    # ========================================================================
    #  Innovation 3: Entropy-based Adaptive Retrieval
    # ========================================================================

    def _compute_feature_entropy(self, feature_vector: np.ndarray) -> float:
        """Compute entropy H of query feature vector.

        H = -sum_i(softmax(f)_i * log(softmax(f)_i))

        Paper Sec 3.3: low entropy = focused query (retrieve few high-relevance items).
        High entropy = ambiguous query (retrieve broader context).
        """
        f = feature_vector.flatten()
        f_tensor = torch.from_numpy(f).float()
        probs = torch.nn.functional.softmax(f_tensor, dim=0)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        entropy = float(-torch.sum(probs * torch.log(probs)).item())

        # Normalize entropy by max possible entropy for this dimensionality
        max_entropy = float(np.log(len(f)))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_entropy

    def search_adaptive(
        self,
        query_text: str,
        query_image: Image.Image,
        device: str = "cuda",
    ) -> Tuple[List[Dict], np.ndarray]:
        """Entropy-based adaptive retrieval (Innovation 3).

        Implements paper Equation (10):
          M^cur = top_k { m_i | (||e_c(q) - e_c(m_i)||_2 < alpha_e * (1 + H(e_c(q))))
                                AND sim(e_c(q), e_c(m_i)) > alpha_s }

        Steps:
          1. Encode query -> e_c(q)
          2. Compute entropy H(e_c(q))
          3. Compute adaptive L2 threshold: alpha_e * (1 + H)
          4. FAISS search with generous k
          5. Filter by L2 distance < threshold
          6. Filter by cosine similarity > alpha_s
          7. Return max_retrieval_num results (in kb format)

        Returns (retrieved_pairs_in_kb_format, indices).
        """
        if len(self.local_entries) == 0:
            return [], np.array([])

        # 1. Encode query
        combined_vec, _, _ = self._encode_entry(query_text, query_image, device)

        # 2. Compute entropy
        H = self._compute_feature_entropy(combined_vec)

        # 3. Adaptive L2 distance threshold
        delta = self.entropy_alpha_e * (1.0 + H)
        logging.info(
            f"Entropy retrieval: H={H:.4f}, adaptive threshold delta={delta:.4f}"
        )

        # 4. FAISS search with generous k
        search_k = min(len(self.local_entries), self.max_retrieval_num * self.entropy_search_multiplier)
        distances, indices = self.index.search(combined_vec, search_k)

        # 5. Filter by L2 distance threshold
        valid_mask = distances[0] < delta
        if not np.any(valid_mask):
            logging.info("  No entries pass adaptive L2 threshold.")
            return [], np.array([])

        filtered_indices = indices[0][valid_mask]

        # 6. Filter by cosine similarity
        query_norm = combined_vec.flatten() / (np.linalg.norm(combined_vec) + 1e-8)
        cosine_sims = []
        for idx in filtered_indices:
            if idx >= len(self.local_entries):
                cosine_sims.append(0.0)
                continue
            mem_feat = self.local_entries[idx].feature_vector.flatten()
            mem_norm = mem_feat / (np.linalg.norm(mem_feat) + 1e-8)
            cosine_sims.append(float(np.dot(query_norm, mem_norm)))

        cosine_sims = np.array(cosine_sims)
        sim_mask = cosine_sims > self.entropy_alpha_s

        if not np.any(sim_mask):
            logging.info("  No entries pass cosine similarity threshold.")
            return [], np.array([])

        final_indices = filtered_indices[sim_mask]
        final_sims = cosine_sims[sim_mask]

        # 7. Sort by cosine similarity (descending), take max_retrieval_num
        sort_order = np.argsort(final_sims)[::-1]
        final_indices = final_indices[sort_order][:self.max_retrieval_num]

        retrieved = [self.local_entries[idx] for idx in final_indices]
        logging.info(
            f"  Retrieved {len(retrieved)} entries (H={H:.4f}, delta={delta:.4f}, "
            f"searched {search_k}, filtered {len(final_indices)})"
        )

        return self._entries_to_kb_format(retrieved), final_indices

    def search(
        self, query_text: str, query_image: Image.Image,
        top_k: int = None, device: str = "cuda"
    ) -> Tuple[List[Dict], np.ndarray]:
        """Backward-compatible search interface.

        If use_entropy_retrieval is True, uses entropy-based adaptive retrieval.
        Otherwise, falls back to legacy fixed-k FAISS search.
        The `top_k` parameter is only used in legacy mode.
        """
        if self.use_entropy_retrieval:
            return self.search_adaptive(query_text, query_image, device)

        # Legacy fixed-k search (preserved for backward compatibility)
        if len(self.local_entries) == 0:
            return [], np.array([])

        if top_k is None:
            top_k = self.max_retrieval_num

        text_inputs = self.preprocess(
            text=query_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.text_embedder.device if hasattr(self.text_embedder, 'device') else device)
        text_vector = (
            self.text_embedder.get_text_features(**text_inputs)
            .cpu().detach().numpy()
        )

        image_inputs = self.preprocess(
            images=query_image, return_tensors="pt", padding=True
        ).to(self.image_model.device if hasattr(self.image_model, 'device') else device)
        image_vector = (
            self.image_model.get_image_features(**image_inputs)
            .cpu().detach().numpy()
        )

        combined_vector = np.concatenate([image_vector, text_vector], axis=1)
        distances, indices = self.index.search(combined_vector, min(top_k, len(self.local_entries)))
        retrieved_pairs = [self.local_entries[i] for i in indices[0] if i != -1 and i < len(self.local_entries)]
        return self._entries_to_kb_format(retrieved_pairs), indices[0]

    # ========================================================================
    #  Main Add Method (Innovations 1 + 2)
    # ========================================================================

    def add(
        self,
        image: Image.Image,
        position: np.ndarray,
        rotation: np.ndarray,
        room_type: str,
        objects: List[Dict],
        caption: str,
        step: int,
        device: str = "cuda",
    ) -> Tuple[int, str]:
        """Add or update a memory entry using viewpoint-aware mechanism.

        This replaces the previous add_to_knowledge_base().

        Args:
            image: PIL Image (current observation)
            position: 3D position vector (habitat coordinates)
            rotation: quaternion [w, x, y, z]
            room_type: room classification string
            objects: list of [{cls, caption, pos}, ...]
            caption: VLM-generated image caption
            step: current exploration step index
            device: torch device

        Returns:
            (entry_id, action) where action is "inserted" or "updated"
        """
        # 1. Build structured text m_i
        structured_text = self._build_structured_text(
            room_type, position, objects, caption
        )

        # 2. Encode features
        combined_vec, image_vec, text_vec = self._encode_entry(
            structured_text, image, device
        )

        # 3. Create candidate entry
        candidate = LocalMemoryEntry(
            entry_id=-1,
            image=image,
            position=np.array(position).flatten(),
            rotation=np.array(rotation).flatten(),
            structured_text=structured_text,
            feature_vector=combined_vec,
            image_vector=image_vec,
            text_vector=text_vec,
            room_type=room_type,
            objects=objects,
            step=step,
        )

        # 4. Viewpoint-aware update logic (Innovation 2)
        if self.use_viewpoint_update and len(self.local_entries) > 0:
            top_k_check = min(
                self.cfg.get("viewpoint_topk_check", 5),
                len(self.local_entries)
            )
            distances, indices = self.index.search(combined_vec, top_k_check)

            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx >= len(self.local_entries):
                    continue
                existing = self.local_entries[idx]
                if self._should_update(candidate, existing):
                    logging.info(
                        f"  Updating entry {existing.entry_id} (FAISS idx {idx}) at step {step}"
                    )
                    self._update_entry(existing, candidate, idx)
                    self._update_global_memory(room_type, objects)
                    return existing.entry_id, "updated"

        # 5. INSERT as new entry
        candidate.entry_id = self.next_entry_id
        self.next_entry_id += 1
        self.index.add(combined_vec)
        self.local_entries.append(candidate)
        self._update_global_memory(room_type, objects)

        logging.info(
            f"  Inserted new entry {candidate.entry_id} at step {step} "
            f"(room={room_type}, objects={[o['cls'] for o in objects]})"
        )
        return candidate.entry_id, "inserted"

    # ========================================================================
    #  Global Memory (Innovation 1 + 4)
    # ========================================================================

    def _update_global_memory(self, room_type: str, objects: List[Dict]):
        """Update the global language-enhanced scene map."""
        if room_type not in self.global_rooms:
            self.global_rooms[room_type] = {
                "objects": set(),
                "visit_count": 0,
                "positions": [],
            }
        room_info = self.global_rooms[room_type]
        room_info["visit_count"] += 1
        for obj in objects:
            room_info["objects"].add(obj["cls"])

    def get_global_spatial_context(
        self, position: np.ndarray, radius: float = 5.0
    ) -> str:
        """Get spatial layout info around a position for planner (Innovation 4).

        Returns text describing nearby rooms and objects to feed into
        the planner's prompt for context-aware navigation decisions.
        """
        pos = np.array(position).flatten()
        nearby_entries = []
        for entry in self.local_entries:
            if entry.position is not None:
                dist = np.linalg.norm(entry.position[:2] - pos[:2])
                if dist < radius:
                    nearby_entries.append(entry)

        rooms_seen = list(dict.fromkeys(
            e.room_type for e in nearby_entries if e.room_type
        ))
        objects_seen = set()
        for e in nearby_entries:
            for obj in e.objects:
                objects_seen.add(obj["cls"])

        context = (
            f"Nearby rooms: {', '.join(rooms_seen) if rooms_seen else 'unknown'}. "
            f"Nearby objects: {', '.join(sorted(objects_seen)) if objects_seen else 'unknown'}. "
            f"Observations near position: {len(nearby_entries)}"
        )
        return context

    def get_room_summary(self) -> str:
        """Get global room-level summary from language-enhanced scene map."""
        if not self.global_rooms:
            return "No rooms explored yet."
        lines = []
        for room_type, info in sorted(self.global_rooms.items()):
            lines.append(
                f"{room_type}: {info['visit_count']} observations, "
                f"objects: {', '.join(sorted(info['objects']))}"
            )
        return "\n".join(lines)

    # ========================================================================
    #  Utility Methods
    # ========================================================================

    def _entries_to_kb_format(
        self, entries: List[LocalMemoryEntry]
    ) -> List[Dict]:
        """Convert LocalMemoryEntry list to legacy kb format for VLM compatibility.

        Returns [{"image": PIL.Image, "text": str}, ...]
        """
        return [{"image": e.image, "text": e.structured_text} for e in entries]

    def clear(self):
        """Reset all memory (called between episodes)."""
        self.index.reset()
        self.local_entries = []
        self.global_rooms = {}
        self.next_entry_id = 0

    def __len__(self):
        return len(self.local_entries)
