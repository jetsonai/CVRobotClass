# CVRobotClass

공유 폴더
https://drive.google.com/drive/folders/1wW3esiUj31kVMBfySVtoFERJdq1btBOn?usp=drive_link

# 추가 셀 내용

result.boxes.cls

"""**NOTE:** YOLO11 can be easily integrated with `supervision` using the familiar `from_ultralytics` connector."""

import supervision as sv

detections = sv.Detections.from_ultralytics(result)

box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

annotated_image = image.copy()

annotated_image = box_annotator.annotate(annotated_image, detections=detections)

annotated_image = label_annotator.annotate(annotated_image, detections=detections)

sv.plot_image(annotated_image, size=(10, 10))
