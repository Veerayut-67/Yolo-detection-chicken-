from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="chicken-detection/train",
    use_segments=False
)