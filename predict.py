from ultralytics import YOLO

model = YOLO('yolov8m_seg-custom.pt')
model.predict(source='Newtest_images/test_img4.png', show=True, save=True, hide_labels=False, hide_conf=False, conf=0.25, save_txt=False, save_crop=False, line_thickness=2)
