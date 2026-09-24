"""Token-alignment extension for CHAIR data annotation.

This module contains ForeSteer-specific logic for COCO 2014 ground-truth
indexing and character-span extraction.
"""

import json
from pathlib import Path

import nltk
import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

from eval_chair import CHAIR, combine_coco_captions


def combine_coco_instances_2014(annotation_path):
    """Load train/val 2014 instances used by the train2014 data pipeline."""
    annotation_path = Path(annotation_path)
    train_path = annotation_path / "instances_train2014.json"
    val_path = annotation_path / "instances_val2014.json"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            "instances_train2014.json and instances_val2014.json are required "
            f"under {annotation_path}"
        )
    with train_path.open("r", encoding="utf-8") as handle:
        train_instances = json.load(handle)
    with val_path.open("r", encoding="utf-8") as handle:
        val_instances = json.load(handle)
    return {
        "categories": train_instances["categories"],
        "annotations": (
            val_instances["annotations"] + train_instances["annotations"]
        ),
    }


class TokenCHAIR(CHAIR):
    """CHAIR extension that returns exact object character spans."""

    def get_annotations(self):
        # Add span rules before the base constructor starts indexing captions.
        self.double_word_dict.setdefault("dining table", "dining table")
        super().get_annotations()

    def caption_to_words(self, caption, return_mentions=False):
        tokenizer = TreebankWordTokenizer()
        spans = list(tokenizer.span_tokenize(caption))
        surface_words = [caption[start:end].lower() for start, end in spans]

        tagged_sent = nltk.pos_tag(surface_words)
        lemmatizer = WordNetLemmatizer()
        words = []
        for word, tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag) or nltk.corpus.wordnet.NOUN
            words.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

        max_phrase_words = max(
            len(phrase.split()) for phrase in self.double_word_dict
        )
        merged_words = []
        merged_spans = []
        index = 0
        while index < len(words):
            matched = False
            max_width = min(max_phrase_words, len(words) - index)
            for width in range(max_width, 1, -1):
                lemma_phrase = " ".join(words[index:index + width])
                surface_phrase = " ".join(surface_words[index:index + width])
                matched_phrase = (
                    lemma_phrase if lemma_phrase in self.double_word_dict
                    else surface_phrase if surface_phrase in self.double_word_dict
                    else None
                )
                if matched_phrase is not None:
                    merged_words.append(self.double_word_dict[matched_phrase])
                    merged_spans.append(
                        (spans[index][0], spans[index + width - 1][1])
                    )
                    index += width
                    matched = True
                    break
            if not matched:
                merged_words.append(words[index])
                merged_spans.append(spans[index])
                index += 1

        if "toilet" in merged_words and "seat" in merged_words:
            kept = [
                (word, span)
                for word, span in zip(merged_words, merged_spans)
                if word != "seat"
            ]
            merged_words = [word for word, _ in kept]
            merged_spans = [span for _, span in kept]

        processed_words = list(merged_words)
        vocabulary = set(self.mscoco_objects)
        object_indices = [
            position
            for position, word in enumerate(merged_words)
            if word in vocabulary
        ]
        object_words = [merged_words[position] for position in object_indices]
        node_words = [self.inverse_synonym_dict[word] for word in object_words]
        mentions = []
        for word, node_word, position in zip(
            object_words, node_words, object_indices
        ):
            start, end = merged_spans[position]
            mentions.append({
                "word": word,
                "node_word": node_word,
                "word_index": position,
                "char_span": [start, end],
                "text": caption[start:end],
            })

        result = (object_words, node_words, object_indices, processed_words)
        if return_mentions:
            return result + (mentions,)
        return result

    def get_annotations_from_segments(self):
        coco_segments = combine_coco_instances_2014(self.coco_path)
        id_to_name = {
            category["id"]: category["name"]
            for category in coco_segments["categories"]
        }
        for annotation in tqdm.tqdm(
            coco_segments["annotations"],
            desc="Indexing COCO segmentation masks",
            unit="mask",
        ):
            image_id = annotation["image_id"]
            category_name = id_to_name[annotation["category_id"]]
            node_word = self.inverse_synonym_dict[category_name]
            self.imid_to_objects[image_id].append(node_word)

    def get_annotations_from_captions(self):
        coco_captions = combine_coco_captions(self.coco_path)
        for annotation in tqdm.tqdm(
            coco_captions["annotations"],
            desc="Indexing COCO ground-truth captions",
            unit="caption",
        ):
            image_id = annotation["image_id"]
            _, node_words, _, _ = self.caption_to_words(annotation["caption"])
            self.imid_to_objects[image_id].extend(node_words)

    def annotate_caption(self, image_id, caption):
        image_id = int(image_id)
        if image_id not in self.imid_to_objects:
            raise KeyError(
                f"COCO image_id={image_id} is absent from the 2014 annotations"
            )

        _, node_words, _, _, mentions = self.caption_to_words(
            caption, return_mentions=True
        )
        gt_objects = self.imid_to_objects[image_id]
        hallucinated_mentions = []
        recalled_objects = set()
        all_mentions = []
        for raw_mention in mentions:
            mention = dict(raw_mention)
            mention["hallucinated"] = mention["node_word"] not in gt_objects
            all_mentions.append(mention)
            if mention["hallucinated"]:
                hallucinated_mentions.append(mention)
            else:
                recalled_objects.add(mention["node_word"])

        recall = (
            len(recalled_objects) / float(len(gt_objects)) if gt_objects else 0.0
        )
        generated_objects = set(node_words)
        precision = (
            len(recalled_objects) / float(len(generated_objects))
            if generated_objects else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        return {
            "mscoco_gt_words": sorted(gt_objects),
            "mscoco_generated_words": node_words,
            "object_mentions": all_mentions,
            "hallucinated_object_mentions": hallucinated_mentions,
            "metrics": {
                "CHAIRs": int(bool(hallucinated_mentions)),
                "CHAIRi": (
                    len(hallucinated_mentions) / float(len(mentions))
                    if mentions else 0.0
                ),
                "Recall": recall,
                "Precision": precision,
                "F1": f1,
            },
        }
