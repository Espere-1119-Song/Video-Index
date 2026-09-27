"""Failure taxonomy of the error attribution.

Thirty judge-facing categories in five groups. Three more categories only appear in outputs:
``sampling_gap`` and ``evidence_not_in_video`` (resolutions of the dense-frame recheck) and
``language_prior_dominated`` (rule 1 of the pre-classification).
"""

# judge-facing categories, in the order they are shown to the judges
CATEGORY_NOTES = [
    ("perception_object_misidentify", "wrong object, person or scene identity"),
    ("perception_fine_grained_action", "subtle action or manipulation missed"),
    ("perception_low_visual_quality", "blur, night or low resolution makes the cue illegible"),
    ("ocr_text", "on-screen text misread or missed"),
    ("temporal_order", "before/after or sequence confusion"),
    ("temporal_localization", "when an event happens is mislocated"),
    ("action_counting", "count of action repetitions wrong"),
    ("speed_duration", "speed or duration misestimated"),
    ("spatial_relation", "left/right, viewpoint, layout, 3D relation"),
    ("physics_causality", "physical plausibility or cause and effect"),
    ("multi_hop_reasoning", "evidence seen, steps not integrated"),
    ("domain_knowledge", "subject knowledge beyond the video is missing"),
    ("language_prior_conflict", "a plausible prior overrode the visual evidence"),
    ("evidence_not_in_input_frames", "the needed evidence is not visible in the frames that were sent"),
    ("audio_needed", "the answer requires the audio track"),
    ("annotation_suspect", "the marked answer looks wrong or inconsistent"),
    ("question_ambiguous", "several defensible readings or answers"),
    ("metric_scale_estimation", "invented metric anchor leads to a wrong absolute distance, size or speed"),
    ("frame_to_timestamp_calibration", "frame index converted to seconds with an assumed duration or frame rate"),
    ("timestamp_frame_anchoring", "an explicit time reference cannot be aligned to the sparse frames"),
    ("subsampled_repetition_aliasing", "periodic action undercounted because the frame sample aliases the cycle"),
    ("task_format_misinterpretation", "output format wrong for the task"),
    ("missing_answer_options", "option texts absent, blank or not rendered in the prompt"),
    ("perception_attribute", "low-level attribute misjudged (colour, shape, size, texture)"),
    ("perception_localization", "wrong region or object anchored"),
    ("perception_comparison", "comparison across elements or frames wrong"),
    ("perception_hallucination", "objects, text or values asserted that are not in the frames"),
    ("perception_object_counting", "count of objects or instances wrong"),
    ("over_refusal", "refusal where an answer was extractable"),
    ("other", "none of the above"),
]
CATEGORIES = [c for c, _ in CATEGORY_NOTES]

GROUP = {}
for _c in ["perception_object_misidentify", "perception_fine_grained_action", "perception_low_visual_quality",
           "ocr_text", "perception_attribute", "perception_localization", "perception_comparison",
           "perception_hallucination", "perception_object_counting"]:
    GROUP[_c] = "perception"
for _c in ["temporal_order", "temporal_localization", "action_counting", "speed_duration",
           "frame_to_timestamp_calibration"]:
    GROUP[_c] = "temporal"
for _c in ["spatial_relation", "physics_causality", "metric_scale_estimation"]:
    GROUP[_c] = "spatial_physical"
for _c in ["multi_hop_reasoning", "domain_knowledge", "language_prior_conflict", "language_prior_dominated"]:
    GROUP[_c] = "reasoning_knowledge"
for _c in ["evidence_not_in_input_frames", "audio_needed", "annotation_suspect", "question_ambiguous", "other",
           "sampling_gap", "evidence_not_in_video", "task_format_misinterpretation",
           "subsampled_repetition_aliasing", "timestamp_frame_anchoring", "missing_answer_options", "over_refusal"]:
    GROUP[_c] = "data_protocol"

DENSE_RESOLUTIONS = ["sampling_gap", "evidence_not_in_video", "audio_needed", "annotation_suspect"]
CAPABILITY_GROUPS = ["perception", "temporal", "spatial_physical", "reasoning_knowledge"]

# ---- grouping of the share tables: fine category -> (group, subgroup)
# evidence_not_in_input_frames is resolved by the dense recheck; `other` is counted apart.
FINE_GROUP = {
    "annotation_suspect": ("annotation", "annotation"),
    "question_ambiguous": ("annotation", "annotation"),
    "audio_needed": ("annotation", "annotation"),
    "missing_answer_options": ("annotation", "annotation"),
    "evidence_not_in_video": ("annotation", "annotation"),
    "sampling_gap": ("coverage", "coverage"),
    "subsampled_repetition_aliasing": ("coverage", "coverage"),
    "timestamp_frame_anchoring": ("coverage", "coverage"),
    "task_format_misinterpretation": ("capability", "reasoning_knowledge"),
    "over_refusal": ("capability", "reasoning_knowledge"),
}
for _c, _g in GROUP.items():
    if _g in CAPABILITY_GROUPS:
        FINE_GROUP[_c] = ("capability", _g)

DENSE_FINAL_GROUP = {
    "sampling_gap": ("coverage", "coverage"),
    "evidence_not_in_video": ("annotation", "annotation"),
    "audio_needed": ("annotation", "annotation"),
    "annotation_suspect": ("annotation", "annotation"),
}
SUBGROUPS = ["annotation", "scoring", "coverage", "perception", "temporal", "spatial_physical", "reasoning_knowledge"]
GROUP_OF_SUB = {"annotation": "annotation", "scoring": "annotation", "coverage": "coverage",
                "perception": "capability", "temporal": "capability", "spatial_physical": "capability",
                "reasoning_knowledge": "capability"}
TIE_ORDER = ["annotation", "coverage", "capability"]
