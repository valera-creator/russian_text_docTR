import torch
import cv2
from doctr.models import crnn_vgg16_bn


def load_image(img_path, device):
    # Чтение изображения
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB

    height = 32
    (h, w) = img.shape[:2]
    ratio = w / h
    width = int(ratio * height)
    img = cv2.resize(img, (width, height))

    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.
    img_tensor = img_tensor.to(device)
    return img_tensor


def get_correct_num_text(text):
    """убирает № в словах, если после них нет цифр, оставляет №, если оно самым последним символом"""
    text = list(text)
    for i in range(len(text)):
        if text[i] == '№' and i + 1 != len(text) and text[i + 1] not in "01234567890":
            text[i] = ' '
    return ''.join(text)


def recognize_text(img_path, device, model, num):
    # получение изображения
    img_tensor = load_image(img_path, device)

    # Распознавание
    with torch.no_grad():
        out = model(img_tensor, return_preds=True)

    # Вывод результатов
    words, _ = zip(*out["preds"])
    text = get_correct_num_text(words[0])

    print(f"Распознанный текст({num}): {text}")
    return words[0]


def load_model(device, text, model_path):
    model = crnn_vgg16_bn(pretrained=False, vocab=text)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main():
    with open('text.txt', encoding='utf-8') as file:
        text = file.readline()

    # Путь к модели
    model_path = r"my_models/crnn_vgg16_bn_20250513-212912.pt"
    model_path2 = r"my_models/crnn_vgg16_bn_20250514-201731.pt"
    model_path3 = r"my_models/crnn_vgg16_bn_20250516-182019.pt"
    model_path4 = r"my_models/crnn_vgg16_bn_20250517-010543.pt"
    model_path5 = r"my_models/crnn_vgg16_bn_20250518-014408.pt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Загрузка моделей
    model = load_model(device, text, model_path)
    model2 = load_model(device, text, model_path2)
    model3 = load_model(device, text, model_path3)
    model4 = load_model(device, text, model_path4)
    model5 = load_model(device, text, model_path5)

    # путь к картинке
    p = r"images/rc7851.png"

    # распознавание
    recognize_text(p, device, model, 1)
    recognize_text(p, device, model2, 2)
    recognize_text(p, device, model3, 3)
    recognize_text(p, device, model4, 4)
    recognize_text(p, device, model5, 5)


if __name__ == "__main__":
    main()
