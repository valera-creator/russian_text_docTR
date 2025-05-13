import torch
import cv2
from doctr.models import crnn_vgg16_bn


def recognize_text(img_path, device, model):
    """картинка с одним словом, по y 40 размером приблизительно"""
    # Чтение изображения
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.
    img_tensor = img_tensor.to(device)

    # Распознавание
    with torch.no_grad():
        out = model(img_tensor, return_preds=True)

    # Вывод результатов
    words, _ = zip(*out["preds"])
    print(f"Распознанный текст: {words[0]}")
    return words[0]


def main():
    with open('text.txt', encoding='utf-8') as file:
        text = file.readline()

    # Путь к модели
    model_path = r"D:\projects\doctr\crnn_vgg16_bn_20250513-212912.pt"

    # Загрузка модели
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = crnn_vgg16_bn(pretrained=False, vocab=text)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    recognize_text(r"C:\Users\valera\Desktop\rc7851.png", device, model)


if __name__ == "__main__":
    main()
