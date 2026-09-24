# Intel Image Classification

Класифікація природних сцен із використанням PyTorch.

## Опис

У проєкті реалізовано два підходи до класифікації зображень із набору **Intel Image Classification**:

1. власна CNN, навчена з нуля;
2. Transfer Learning на основі попередньо навченої ResNet18.

Класи датасету:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

## Дані

Структура даних:

```text
datasets/
├── seg_train/seg_train/
├── seg_test/seg_test/
└── seg_pred/seg_pred/
```

Навчальна вибірка додатково розділяється на:

- train — 80%;
- validation — 20%.

Тестова вибірка використовується лише для фінального оцінювання.

## Реалізація

### SimpleCNN

Власна CNN містить:

- три згорткові блоки;
- `Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`;
- `AdaptiveAvgPool2d`;
- повнозв’язні шари;
- `Dropout`.

### ResNet18

Для Transfer Learning використано ResNet18 із попередньо навченими вагами ImageNet. Базові шари заморожено, а фінальний класифікатор замінено на блок для шести класів.

## Метрики

Якість моделей оцінюється за допомогою:

- accuracy;
- F1-score macro;
- F1-score weighted;
- precision і recall;
- confusion matrix.

Основною метрикою вибору найкращої моделі є `F1-macro`.

## Результати

Найкращі валідаційні результати:

| Модель | Accuracy | F1-macro |
|---|---:|---:|
| SimpleCNN | 82.15% | 0.8232 |
| ResNet18 Transfer Learning | 90.67% | 0.9083 |

ResNet18 показала кращий результат завдяки використанню попередньо навчених ознак ImageNet.

## Встановлення

Рекомендовано використовувати віртуальне середовище Python.

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy matplotlib seaborn scikit-learn scipy pandas pillow tqdm ipykernel jupyter
```

Для роботи Notebook зареєструйте Kernel:

```powershell
python -m ipykernel install --user --name goit-dlcv-gpu --display-name "Python (goit-dlcv-gpu)"
```

## Запуск

1. Відкрийте `hw06.ipynb` у VS Code або Jupyter Notebook.
2. Виберіть Kernel `Python (goit-dlcv-gpu)`.
3. Переконайтеся, що CUDA доступна.
4. Послідовно виконайте комірки ноутбука.
5. Перегляньте метрики, classification report, матрицю помилок і візуалізацію прогнозів.

Перевірка CUDA:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
```

## Автор

Євген Петров
