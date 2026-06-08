from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')

    results = model.train(
        data='/home/max/Стільниця/human_dec_yolo8_new/data.yaml', 
        
        epochs=100,
        
        imgsz=832,
        batch=4,    
        
        name='yolov8s_human_detect', 
        
        device=''   # Автоматично вибере GPU (якщо доступно) або процесор
    )

if __name__ == '__main__':   ''
    main()