# pip install "python-doctr[torch,viz,html,contrib]"
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

doc = DocumentFile.from_images("test.png")
model = ocr_predictor(pretrained=True)
result = model(doc)
print(result.export())
