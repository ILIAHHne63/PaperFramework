# Модель для аугментации изображений

Модель для автоматической аугментации изображений
## 🛠 Установка

1. Клонируйте репозиторий, установите окружение, скачайте нужные зависимости и веса:
```bash
git clone https://github.com/ILIAHHne63/PaperFramework
cd PaperFramework

conda create --name aug_model python=3.10
conda activate aug_model

pip install -r requirements.txt
cp -r src/alpha-clip/* "AugModel/models/AlphaCLIP/"
bash checkpoints.sh
```
Также скачайте файл по данной ссылке: https://drive.google.com/file/d/1e83wWQh9Tsficx0HM2rmc2KyYfBMpumf/view, положите данный файл в AugmentationModel/AugModel/models/checkpoints/

Далее используйте Example.ipynb
