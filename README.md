# Polariton Condensate

CPU-модель двумерного открытого уравнения Гросса–Питаевского с одним
экситонным резервуаром. Постановка и критерии приёмки находятся в
[ТЗ v3](ТЗ_поляритонный_конденсат_2D_компактная_архитектура_v3.md).
Гауссово пятно и кольцо являются нашими геометриями исследования, а не
воспроизведением решётки из статьи Alyatkin et al. (2024).

Главный интерфейс — [polariton_condensate.ipynb](polariton_condensate.ipynb).
Длительные ячейки выключены флагами DO_*; открытие ноутбука не запускает
серии расчётов. model.py содержит уравнения и единицы, numerics.py — FFT
решатель, diagnostics.py — наблюдаемые, storage.py — конфигурации и данные
прогона, plotting.py — рисунки из сохранённых массивов.

## Установка и быстрые проверки

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/jupyter lab polariton_condensate.ipynb
~~~

Пример из Python:

~~~python
from storage import load_config
from numerics import run_validation_suite, benchmark, simulate_2d

config = load_config("configs/gaussian.yaml")
print(run_validation_suite(config))
print(benchmark(config, steps=300, sizes=(256, 512)))
result = simulate_2d(config)
print(result["path"], result["summary"]["run_status"])
~~~

Конфигурации gaussian.yaml и ring.yaml наследуют единственный base.yaml.
Каждый расчёт получает собственный каталог results/<run_id>/ с полным
конфигом, снимками .npz, контрольными точками, временными рядами, журналом
предупреждений и итоговой диагностикой. Метаданные содержат оценку памяти,
размера выходных файлов и свободного места перед запуском. Результаты не входят в Git.

RUN_INCOMPLETE означает остановку с контрольной точкой. Продолжение:
simulate_2d(config, resume_dir=run_path). Допускается увеличить только
time.t_final_ps; остальные параметры должны совпасть. Завершённый расчёт тоже
можно продлить, указав более позднее конечное время. При Ctrl+C расчёт
сохраняет RUN_INCOMPLETE и продолжает с последней контрольной точки.
Рисунки строятся
через plotting.make_figures без нового 2D-расчёта. Для F5 можно передать
дополнительные сохранённые прогоны в gaussian_spectrum_runs и
ring_spectrum_runs; все панели используют общую цветовую шкалу.

По умолчанию граница периодическая. Если поле доходит до неё, можно явно
включить численный поглощающий слой через space.absorbing_layer=true;
space.absorber_width_um и space.absorber_gamma_max_per_ps задают его профиль.
Его потери входят в интегральный баланс. Такой прогон помечается в метаданных
и проверяется сравнением с большей периодической областью.

P0=1 — однородный ориентир, а не порог локального источника. Функция
find_threshold_2d измеряет линейный рост на замороженной нулевой ветви;
PRELIMINARY требует независимой проверки сетки и области. NOT_TESTED не
означает успешного прохождения проверки.
