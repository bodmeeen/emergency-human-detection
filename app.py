import cv2
import gradio as gr
from ultralytics import YOLO

model = YOLO('runs/detect/yolov8s_human_detect/weights/best.pt')

def process_image(input_img):
    # Оскільки Gradio передає зображення в форматі RGB, 
    # а OpenCV та YOLO зазвичай працюють з BGR, робимо конвертацію
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    
    results = model.predict(source=img_bgr, conf = 0.20)
    
    # Отримуємо картинку з уже намальованими синіми рамками
    annotated_img_bgr = results[0].plot()
    
    # Конвертуємо назад в RGB для коректного відображення в браузері
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    return annotated_img_rgb

interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Вхідне зображення (фото з дрона або місця пожежі)"),
    outputs=gr.Image(label="Результат детекції людей"),
    title="Система пошуку людей в умовах надзвичайних ситуацій",
    description="Завантажте фотографію. YOLOv8 проаналізує кадр та виділить силуети людей рамками.",
)

if __name__ == "__main__":
    interface.launch()