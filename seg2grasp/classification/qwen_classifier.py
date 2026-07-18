"""Open-vocabulary object classification with a Qwen vision-language model.

This is the Seg2Grasp classification module. It replaces the paper's fine-tuned
Mask-CLIP with a modern VLM: after segmentation selects a target object, the
classifier is shown the **full scene image** (for context) and the **cropped
target object**, and asked to name the target's category from a fixed list
(``configs/categories.yaml``, the paper's product categories).

The Qwen model is loaded in-process via transformers (SDPA attention), once per
:class:`QwenClassifier`, and reused across calls.
"""
import json

from .. import paths


# --------------------------------------------------------------------------- helpers
def _slug(category):
    """Category -> lowercase underscore slug."""
    return category.strip().lower().replace(" ", "_")


def _bgr_to_pil(bgr):
    """OpenCV BGR ndarray -> RGB PIL.Image; None on empty input."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return None
    from PIL import Image
    return Image.fromarray(bgr[:, :, ::-1].copy())


def _extract_json(text):
    """Parse the first JSON object found in ``text`` (model may wrap it in prose)."""
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _match_category(text, categories):
    """Resolve a model answer to one of ``categories`` (exact/slug/substring)."""
    if not text:
        return None
    t = str(text).strip().lower()
    for c in categories:
        if t == c.lower() or t == _slug(c) or t.replace("_", " ") == c.lower():
            return c
    for c in categories:
        if c.lower() in t or _slug(c) in t.replace(" ", "_"):
            return c
    return None


def _last_category(text, categories):
    """Category mentioned last in ``text`` (fallback when JSON doesn't parse)."""
    if not text:
        return None
    t = text.lower()
    best, pos = None, -1
    for c in categories:
        p = max(t.rfind(c.lower()), t.rfind(_slug(c)))
        if p > pos:
            pos, best = p, c
    return best


def load_categories(config_path=None):
    """Load the category list from a YAML file (key ``categories``)."""
    config_path = config_path or paths.CATEGORIES_CONFIG
    import yaml
    with open(config_path) as f:
        data = yaml.safe_load(f)
    cats = data.get("categories", data) if isinstance(data, dict) else data
    return list(cats)


# --------------------------------------------------------------------------- classifier
class QwenClassifier:
    """Classify a segmented target object into one of ``categories`` using Qwen-VL.

    Args:
        categories (list[str], optional): candidate labels; defaults to
            ``configs/categories.yaml``.
        descriptions (dict, optional): short distinguishing hint per category,
            shown next to each option (improves accuracy). ``None`` to disable.
        model_path (str, optional): model selection — an alias in
            ``paths.QWEN_MODELS`` (e.g. "27b-fp8", "35b"), a full HF repo id, or a
            local weights dir. Defaults to ``paths.QWEN_DEFAULT_MODEL``.
        dtype (str): "auto" (honor the checkpoint's dtype/quant, e.g. FP8) or a
            torch dtype name like "bfloat16".
        open_vocab (bool): if True, let the model name the object freely instead
            of restricting to ``categories`` (off by default; the paper uses a
            fixed list).
    """

    def __init__(self, categories=None, descriptions=None, *, model_path=None,
                 dtype="auto", open_vocab=False, thinking=False, max_new_tokens=1024,
                 temperature=0.6, top_p=0.95, top_k=20, device=None, logger=None):
        self.categories = list(categories) if categories is not None else load_categories()
        self.descriptions = descriptions or {}
        self.dtype = dtype
        self.open_vocab = open_vocab
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.log = logger
        self._load_model(model_path, device)

    # ---- model ----
    def _load_model(self, model_path, device):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self._torch = torch
        model_path = paths.resolve_qwen_model(model_path)   # alias / repo id / local path
        # "auto" lets transformers honor the checkpoint's own dtype/quant (e.g. FP8);
        # otherwise map a name like "bfloat16" to the torch dtype.
        dtype = self.dtype if self.dtype == "auto" else getattr(torch, self.dtype)
        if self.log:
            self.log.info(f"QwenClassifier: loading {model_path} (sdpa, dtype={self.dtype}) ...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation="sdpa",   # torch kernels; avoids precompiled flash-attn PTX issues
            device_map=(device or "auto"))
        self.model.eval()

    # ---- generation ----
    @staticmethod
    def _img(pil):
        return {"type": "image", "image": pil}

    def _generate(self, content, thinking=None):
        torch = self._torch
        think = self.thinking if thinking is None else thinking
        messages = [{"role": "user", "content": content}]
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", enable_thinking=think).to(self.model.device)
        except TypeError:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self.model.device)
        max_new = max(self.max_new_tokens, 8192) if think else self.max_new_tokens
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True,
                temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        gen = out[0][inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(gen, skip_special_tokens=True)
        if "</think>" in text:
            text = text.split("</think>")[-1]
        return text.strip()

    def _build_content(self, scene_pil, crop_pil):
        """Full scene (context) + target crop + category options + JSON instruction."""
        content = [{"type": "text", "text":
            "You classify the target object in a robotic bin-picking scene. "
            "First, here is the FULL scene image for context:"}]
        if scene_pil is not None:
            content.append(self._img(scene_pil))
        content.append({"type": "text", "text":
            "The robot is about to grasp ONE target object. Here is a CLOSE-UP crop of that "
            "target object:"})
        content.append(self._img(crop_pil))

        if self.open_vocab:
            content.append({"type": "text", "text":
                "Name the target object's product category with a short noun phrase. "
                'Answer with ONLY this JSON object:\n{"answer": "<category>", "confidence": <0-1>}'})
        else:
            opts = "\n".join(
                f'  - "{c}"' + (f": {self.descriptions[c]}" if self.descriptions.get(c) else "")
                for c in self.categories)
            content.append({"type": "text", "text":
                "Classify the TARGET object (the crop) into exactly ONE of these categories:\n" + opts +
                "\n\nDo NOT explain. Answer IMMEDIATELY with ONLY this single JSON object:\n"
                '{"answer": "<one of the category names above, verbatim>", "confidence": <number 0-1>}'})
        return content

    def classify(self, scene_bgr, crop_bgr):
        """Classify the target ``crop_bgr`` given the ``scene_bgr`` context.

        Returns ``{class_name, class_idx, confidence, raw}``. ``class_name`` is
        ``None`` if nothing parseable came back. In open-vocab mode ``class_idx``
        is -1 and ``class_name`` is the free-form answer.
        """
        result = {"class_name": None, "class_idx": -1, "confidence": None, "raw": ""}
        crop = _bgr_to_pil(crop_bgr)
        if crop is None:
            if self.log:
                self.log.warning("QwenClassifier.classify: empty/invalid crop.")
            return result
        scene = _bgr_to_pil(scene_bgr)

        answer = self._generate(self._build_content(scene, crop))
        result["raw"] = answer
        obj = _extract_json(answer)
        conf = obj.get("confidence") if isinstance(obj, dict) else None
        if isinstance(conf, (int, float)):
            result["confidence"] = float(conf)

        if self.open_vocab:
            result["class_name"] = (obj.get("answer") if isinstance(obj, dict) else None) or answer
            return result

        cat = _match_category(obj.get("answer"), self.categories) if isinstance(obj, dict) else None
        if cat is None:
            cat = _last_category(answer, self.categories)
        if cat is not None:
            result["class_name"] = cat
            result["class_idx"] = self.categories.index(cat)
        elif self.log:
            self.log.warning(f"QwenClassifier: could not parse a category from: {answer!r}")
        return result
