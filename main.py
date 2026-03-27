import cv2
import argparse
import pandas as pd
import os
from ultralytics import YOLO
import numpy as np

class TableOccupancyDetector:
    def __init__(self, video_path, roi=None):
        """
        Инициализация детектора occupancy столика
        
        Args:
            video_path: путь к видеофайлу
            roi: координаты области столика (x, y, w, h), если None - выбрать вручную
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Получаем параметры видео
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Загружаем YOLO модель для детекции людей
        self.model = YOLO('yolov8n.pt')  # Используем предобученную модель
        
        # Определяем область столика
        if roi is None:
            self.roi = self.select_roi()
        else:
            self.roi = roi
            
        # Состояния
        self.current_state = "empty"  # empty, occupied
        self.events = []  # Список событий
        self.last_empty_time = None  # Время когда стол стал пустым
        self.last_occupied_time = None  # Для отслеживания коротких периодов

        # Пороги для детекции
        self.occupancy_threshold = 0.3  # Порог уверенности для детекции человека
        self.min_frames_for_occupancy = 15  # Минимальное количество кадров для принятия сотояния занят
        self.min_frames_for_empty = 15  # Минимальное количество кадров для принятия сотояния пуст
        self.min_occupied_duration = 3.0  # Минимальная длительность "занятого" состояния (секунд)
        self.min_empty_duration = 10.0  # Минимальная длительность "пустого" состояния (секунд)

        # Счетчики для сглаживания
        self.occupied_counter = 0
        self.empty_counter = 0

    def format_time(self, seconds):
        """
        Преобразует секунды в формат ЧЧ:ММ:СС.мс
        """

        # Проверка на NaN или None
        if seconds is None or np.isnan(seconds):
            return "N/A"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        else:
            return f"{minutes:02d}:{secs:06.3f}"
        
    def select_roi(self):
        """
        Интерактивный выбор области столика на первом кадре
        """
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать видео")
            
        # Показываем первый кадр для выбора ROI
        cv2.namedWindow("Выберите область столика", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Выберите область столика", 1280, 720)
        
        # Выбираем ROI
        roi = cv2.selectROI("Выберите область столика", frame, False, False)
        cv2.destroyWindow("Выберите область столика")
        
        # Сбрасываем видео на начало
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return roi
    
    def detect_people_in_roi(self, frame):
        """
        Детекция людей в области столика
        
        Returns:
            bool: True если есть человек в ROI, False если нет
        """
        x, y, w, h = self.roi
        roi_frame = frame[y:y+h, x:x+w]
        
        if roi_frame.size == 0:
            return False
            
        # Детекция людей с помощью YOLO
        results = self.model(roi_frame, verbose=False, classes=[0])
        
        # Проверяем детекции
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                # Проверяем уверенность
                for box in r.boxes:
                    if float(box.conf[0]) > self.occupancy_threshold:
                        return True
        return False
    
    def update_state(self, is_occupied, frame_idx, timestamp_sec):
        """
        Обновление состояния столика с учетом сглаживания и минимальной длительности
        """
        # Обновляем счетчики для сглаживания
        if is_occupied:
            self.occupied_counter += 1
            self.empty_counter = 0
        else:
            self.empty_counter += 1
            self.occupied_counter = 0
        
        changed = False
        
        # Проверяем нужно ли сменить состояние
        if self.current_state == "empty":
            if self.occupied_counter >= self.min_frames_for_occupancy:
                # Проверяем, не слишком ли коротким было "пустое" состояние
                if self.last_empty_time is not None:
                    empty_duration = timestamp_sec - self.last_empty_time
                    
                    # Если "пустое" состояние было слишком коротким, игнорируем его
                    if empty_duration < self.min_empty_duration:
                        # Откатываем последнее событие "empty"
                        if self.events and self.events[-1]['event'] == 'empty':
                            self.events.pop()
                            print(f"  → Игнорируем короткое освобождение ({empty_duration:.1f} сек)")
                        
                        # Не меняем состояние, продолжаем считать стол занятым
                        self.current_state = "occupied"
                        self.occupied_counter = self.min_frames_for_occupancy
                        self.empty_counter = 0
                        self.last_empty_time = None
                        return False
                
                # Нормальная смена состояния
                self.current_state = "occupied"
                
                # Создаем событие
                event = {
                    'timestamp_sec': timestamp_sec,
                    'timestamp_formatted': self.format_time(timestamp_sec),
                    'frame': frame_idx,
                    'event': 'occupied'
                }
                
                # Вычисляем время подхода
                if self.last_empty_time is not None:
                    approach_time = timestamp_sec - self.last_empty_time
                    event['approach_time_sec'] = approach_time
                    event['approach_time_formatted'] = self.format_time(approach_time)
                    print(f"[{self.format_time(timestamp_sec)}] Стол ЗАНЯТ (подход через {self.format_time(approach_time)})")
                else:
                    print(f"[{timestamp_sec:.1f} сек] Стол ЗАНЯТ")
                
                self.events.append(event)
                changed = True
                
        elif self.current_state == "occupied":
            if self.empty_counter >= self.min_frames_for_empty:
                # ПРОВЕРКА: не слишком ли коротким было "занятое" состояние
                if self.last_occupied_time is not None:
                    occupied_duration = timestamp_sec - self.last_occupied_time
                    
                    # Если "занятое" состояние было слишком коротким, игнорируем его
                    if occupied_duration < self.min_occupied_duration:
                        # Откатываем последнее событие "occupied"
                        if self.events and self.events[-1]['event'] == 'occupied':
                            self.events.pop()
                            print(f"  → Игнорируем короткое занятие ({occupied_duration:.1f} сек)")
                        
                        # Не меняем состояние, продолжаем считать стол пустым
                        self.current_state = "empty"
                        self.empty_counter = self.min_frames_for_empty
                        self.occupied_counter = 0
                        self.last_occupied_time = None
                        return False
                
                # Нормальная смена состояния
                self.current_state = "empty"
                
                # Создаем событие
                event = {
                    'timestamp_sec': timestamp_sec,
                    'timestamp_formatted': self.format_time(timestamp_sec),
                    'frame': frame_idx,
                    'event': 'empty'
                }
                self.events.append(event)
                self.last_empty_time = timestamp_sec
                print(f"[{self.format_time(timestamp_sec)}] Стол ПУСТ")
                changed = True
        
        return changed
    
    def get_state_color(self):
        """
        Возвращает цвет для bounding box в зависимости от состояния
        """
        if self.current_state == "empty":
            return (0, 255, 0)  # Зеленый - пусто
        else:
            return (0, 0, 255)  # Красный - занято
    
    def draw_roi(self, frame):
        """
        Рисует bounding box столика на кадре
        """
        x, y, w, h = self.roi
        color = self.get_state_color()
        
        # Рисуем прямоугольник
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Добавляем текст с состоянием
        state_text = "EMPTY" if self.current_state == "empty" else "OCCUPIED"
        cv2.putText(frame, state_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Показываем время в удобном формате
        current_time = self.frame_count / self.fps
        time_str = self.format_time(current_time)
        cv2.putText(frame, f"Time: {time_str}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self, output_path='output.mp4'):
        """
        Основной цикл обработки видео
        """
        print(f"Начинаем обработку видео: {self.video_path}")
        print(f"Всего кадров: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Область столика: {self.roi}")
        print(f"Минимальная длительность 'пустого' состояния: {self.min_empty_duration} сек\n")
        
        # Создаем VideoWriter для сохранения результата
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                            (self.width, self.height))
        
        self.frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp_sec = self.frame_count / self.fps
            
            # Детектируем людей в ROI
            has_people = self.detect_people_in_roi(frame)
            
            # Обновляем состояние
            self.update_state(has_people, self.frame_count, timestamp_sec)
            
            # Визуализируем
            frame_with_roi = self.draw_roi(frame)
            
            # Сохраняем кадр
            out.write(frame_with_roi)
            
            # Выводим прогресс
            if self.frame_count % 100 == 0:
                progress = (self.frame_count / self.total_frames) * 100
                print(f"Прогресс: {progress:.1f}%")
                
            self.frame_count += 1
            
        # Освобождаем ресурсы
        self.cap.release()
        out.release()
        
        print(f"\nОбработка завершена. Результат сохранен в {output_path}")
        
        return self.generate_report()
    
    def generate_report(self):
        """
        Генерирует отчет с аналитикой
        """
        if not self.events:
            print("Нет зарегистрированных событий")
            return None
            
        # Создаем DataFrame с событиями
        df = pd.DataFrame(self.events)
        
        # Отладка: показываем все события
        print("\n" + "="*60)
        print("ВСЕ СОБЫТИЯ:")
        print(df.to_string())
        print("="*60)
    
        # Собираем время подходов
        approach_times = []
        for i, row in df.iterrows():
            if row['event'] == 'occupied':
                if 'approach_time_sec' in row and pd.notna(row['approach_time_sec']):
                    approach_times.append(row['approach_time_sec'])
                    print(f"  Найден подход: {row['approach_time_sec']:.1f} сек")
                else:
                    print(f"  Событие occupied без approach_time_sec (индекс {i})")
        
        print("\n" + "="*60)
        print("ОТЧЕТ ПО АНАЛИТИКЕ")
        print("="*60)
        print(f"Всего событий: {len(self.events)}")
        print(f"Событий 'стол занят': {len(df[df['event']=='occupied'])}")
        print(f"Событий 'стол пуст': {len(df[df['event']=='empty'])}")

        # Вычисляем статистику
        if approach_times:
            avg_time = np.mean(approach_times)
            min_time = np.min(approach_times)
            max_time = np.max(approach_times)
            
            print(f"\nСтатистика времени между уходом и следующим подходом:")
            print(f"  Среднее: {self.format_time(avg_time)}")
            print(f"  Минимальное: {self.format_time(min_time)}")
            print(f"  Максимальное: {self.format_time(max_time)}")
            print(f"  Количество замеров: {len(approach_times)}")
            print("="*60)
            
            # Сохраняем отчет
            report_file = f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("Отчет по детекции уборки столиков\n")
                f.write("="*60 + "\n")
                f.write(f"Видео: {self.video_path}\n")
                f.write(f"Область столика: {self.roi}\n")
                f.write(f"Всего событий: {len(self.events)}\n")
                f.write(f"Среднее время между уходом и подходом: {self.format_time(avg_time)}\n")
                f.write(f"Минимальное время: {self.format_time(min_time)}\n")
                f.write(f"Максимальное время: {self.format_time(max_time)}\n")
                f.write(f"\nДетали событий:\n")
                f.write(df.to_string())
                
            print(f"\nДетальный отчет сохранен в {report_file}")
            
            return {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'total_events': len(self.events)
            }
        else:
            print("Не найдено полных циклов 'empty -> occupied'(пустой->занятый)")
            print("="*60)
            return None

def main():
    parser = argparse.ArgumentParser(description='Детектор occupancy столика по видео')
    parser.add_argument('--video', type=str, required=True,
                       help='Путь к видеофайлу')
    parser.add_argument('--roi', type=str, default=None,
                       help='Координаты ROI в формате "x,y,w,h"')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='Путь для сохранения результата')
    
    args = parser.parse_args()
    
    # Проверяем существование видео
    if not os.path.exists(args.video):
        print(f"Ошибка: видеофайл {args.video} не найден")
        return
    
    # Парсим ROI если передан
    roi = None
    if args.roi:
        try:
            x, y, w, h = map(int, args.roi.split(','))
            roi = (x, y, w, h)
        except:
            print("Неверный формат ROI. Используйте: x,y,w,h")
            return
    
    # Создаем детектор и обрабатываем видео
    detector = TableOccupancyDetector(args.video, roi)
    results = detector.process_video(args.output)
    
    if results:
        print(f"\nРезультат сохранен в {args.output}")
        print(f"Среднее время между уходом и подходом: {detector.format_time(results['avg_time'])}")

if __name__ == "__main__":
    main()