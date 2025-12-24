import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class AdvancedObjectTracker:
    def __init__(self):
        # Kamera
        self.cap = cv2.VideoCapture(0)
        
        # Renk aralıkları (HSV)
        self.color_ranges = {
            "Kirmizi": [([0, 150, 50], [10, 255, 255]), ([170, 150, 50], [180, 255, 255])],
            "Yesil": [([40, 50, 50], [80, 255, 255])],
            "Mavi": [([100, 150, 50], [140, 255, 255])],
            "Sari": [([20, 100, 100], [30, 255, 255])],
            "Turuncu": [([10, 100, 100], [20, 255, 255])],
            "Pembe": [([140, 50, 50], [170, 255, 255])]
        }
        
        # Takip edilen renk
        self.tracked_color = "Mavi"
        
        # Hareket takibi için
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25)
        self.motion_history = deque(maxlen=20)  # Son 20 pozisyon
        
        # El tespiti için - farklı cascade dosyalarını dene
        self.hand_cascade = None
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_hand.xml',
            cv2.data.haarcascades + 'haarcascade_palm.xml',
            'hand_cascade.xml'  # Eğer indirdiysen
        ]
        
        for path in cascade_paths:
            try:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self.hand_cascade = cascade
                    print(f"El cascade yüklendi: {path}")
                    break
            except:
                continue
        
        if self.hand_cascade is None:
            print("El cascade yüklenemedi, alternatif yöntem kullanılacak.")
        
        # UI için
        self.selected_roi = None
        self.tracking_mode = "color"  # color, motion, hand, custom
        self.show_info = True
        
    def get_color_mask(self, hsv_frame, color_name):
        """Belirtilen renk için maske oluştur"""
        masks = []
        
        if color_name in self.color_ranges:
            for lower, upper in self.color_ranges[color_name]:
                lower_np = np.array(lower)
                upper_np = np.array(upper)
                mask = cv2.inRange(hsv_frame, lower_np, upper_np)
                masks.append(mask)
        
        if masks:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Gürültü temizleme
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
            
            return combined_mask
        
        return None
    
    def detect_hand(self, frame, gray_frame):
        """El tespiti yapar"""
        hands = []
        
        if self.hand_cascade is not None:
            try:
                # Cascade ile el tespiti
                detected_hands = self.hand_cascade.detectMultiScale(
                    gray_frame, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(50, 50)
                )
                
                for (x, y, w, h) in detected_hands:
                    # El bölgesini işle
                    hand_roi = frame[y:y+h, x:x+w]
                    
                    # Deri rengi tespiti (HSV)
                    hsv_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
                    skin_lower = np.array([0, 30, 60])
                    skin_upper = np.array([20, 150, 255])
                    skin_mask = cv2.inRange(hsv_roi, skin_lower, skin_upper)
                    
                    # El şekli kontrolü
                    skin_ratio = np.sum(skin_mask > 0) / (w * h)
                    
                    if skin_ratio > 0.3:  # Yeterince deri rengi varsa
                        hands.append((x, y, w, h, skin_ratio))
            except:
                pass
        
        # Alternatif yöntem: renk bazlı el tespiti
        if not hands:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Deri rengi aralığı
            skin_lower = np.array([0, 30, 60])
            skin_upper = np.array([20, 150, 255])
            skin_mask = cv2.inRange(hsv_frame, skin_lower, skin_upper)
            
            # Morfolojik işlemler
            kernel = np.ones((5, 5), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
            
            # Konturları bul
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Büyük alanlar (el olabilir)
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # En-boy oranı kontrolü
                    aspect_ratio = w / float(h)
                    if 0.5 < aspect_ratio < 2.0:  # El benzeri oran
                        hands.append((x, y, w, h, 0.5))
        
        return hands
    
    def detect_clothing_color(self, frame, x, y, w, h):
        """Kıyafet rengini tespit eder"""
        # Üst bölge (göğüs/kıyafet bölgesi)
        clothing_roi = frame[max(0, y + h//2):y + h, x:x + w]
        
        if clothing_roi.size == 0:
            return "Belirlenemedi"
        
        # HSV'ye çevir
        hsv_roi = cv2.cvtColor(clothing_roi, cv2.COLOR_BGR2HSV)
        
        # En baskın rengi bul
        hist_h = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        
        dominant_hue = np.argmax(hist_h)
        
        # Renk ismini belirle
        color_name = "Belirlenemedi"
        
        if 0 <= dominant_hue < 10 or 170 <= dominant_hue < 180:
            color_name = "Kirmizi"
        elif 10 <= dominant_hue < 25:
            color_name = "Turuncu"
        elif 25 <= dominant_hue < 35:
            color_name = "Sari"
        elif 35 <= dominant_hue < 85:
            color_name = "Yesil"
        elif 85 <= dominant_hue < 130:
            color_name = "Mavi"
        elif 130 <= dominant_hue < 170:
            color_name = "Pembe"
        
        return color_name
    
    def track_by_color(self, frame, hsv_frame):
        """Renk bazlı takip"""
        mask = self.get_color_mask(hsv_frame, self.tracked_color)
        
        if mask is None:
            return None
        
        # Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # En büyük konturu bul
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Minimum alan kontrolü
            if cv2.contourArea(largest_contour) > 500:
                # Sınırlayıcı kutu
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Merkez nokta
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Hareket geçmişine ekle
                self.motion_history.append((center_x, center_y))
                
                return {
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': cv2.contourArea(largest_contour),
                    'color': self.tracked_color,
                    'mask': mask
                }
        
        return None
    
    def track_by_motion(self, frame, gray_frame):
        """Hareket bazlı takip"""
        # Hareket maskesi
        fgmask = self.bg_subtractor.apply(gray_frame)
        
        # Gürültü temizleme
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        
        # Konturları bul
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # En büyük hareket bölgesini bul
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 1000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                self.motion_history.append((center_x, center_y))
                
                return {
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': cv2.contourArea(largest_contour),
                    'type': 'hareket',
                    'mask': fgmask
                }
        
        return None
    
    def draw_tracking_info(self, frame, track_info):
        """Takip bilgilerini çizer"""
        if track_info is None:
            return frame
        
        x, y, w, h = track_info['bbox']
        center_x, center_y = track_info['center']
        
        # Çerçeve çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Merkez noktası
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Bilgi yazıları
        info_y = 30
        cv2.putText(frame, f"MOD: {self.tracking_mode.upper()}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if 'color' in track_info:
            cv2.putText(frame, f"RENK: {track_info['color']}", (10, info_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"X:{center_x}, Y:{center_y}", (10, info_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"ALAN: {track_info['area']:.0f}", (10, info_y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Hareket yolu
        if len(self.motion_history) > 1:
            for i in range(1, len(self.motion_history)):
                cv2.line(frame, self.motion_history[i-1], self.motion_history[i],
                        (0, 255, 255), 2)
        
        return frame
    
    def draw_color_palette(self, frame):
        """Renk paletini çizer"""
        palette_height = 50
        palette_width = frame.shape[1]
        palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
        
        num_colors = len(self.color_ranges)
        color_width = palette_width // num_colors
        
        for i, (color_name, ranges) in enumerate(self.color_ranges.items()):
            x_start = i * color_width
            x_end = (i + 1) * color_width
            
            # Renk belirle
            if color_name == "Kirmizi":
                color = (0, 0, 255)
            elif color_name == "Yesil":
                color = (0, 255, 0)
            elif color_name == "Mavi":
                color = (255, 0, 0)
            elif color_name == "Sari":
                color = (0, 255, 255)
            elif color_name == "Turuncu":
                color = (0, 165, 255)
            elif color_name == "Pembe":
                color = (255, 0, 255)
            else:
                color = (255, 255, 255)
            
            # Renk çubuğunu çiz
            cv2.rectangle(palette, (x_start, 0), (x_end, palette_height), color, -1)
            
            # Seçili renk için çerçeve
            if color_name == self.tracked_color:
                cv2.rectangle(palette, (x_start, 0), (x_end, palette_height), (255, 255, 255), 3)
            
            # Renk adını yaz
            text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x_start + (color_width - text_size[0]) // 2
            cv2.putText(palette, color_name, (text_x, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Palette'yi frame'in üstüne ekle
        frame[:palette_height, :] = palette
        
        return frame
    
    def run(self):
        """Ana döngü"""
        print("="*60)
        print("NESNE TAKİP SİSTEMİ")
        print("="*60)
        print("KOMUTLAR:")
        print("  SPACE: Mod değiştir (renk/hareket/el)")
        print("  1-6: Renk seç (1:Kırmızı, 2:Yeşil, 3:Mavi, 4:Sarı, 5:Turuncu, 6:Pembe)")
        print("  i: Bilgileri aç/kapat")
        print("  s: Ekran görüntüsü al")
        print("  q: Çıkış")
        print("="*60)
        
        screenshot_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Aynalı görüntü
            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()
            
            # Gri ton
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            track_info = None
            
            # Moda göre takip
            if self.tracking_mode == "color":
                track_info = self.track_by_color(frame, hsv)
                
            elif self.tracking_mode == "motion":
                track_info = self.track_by_motion(frame, gray)
                
            elif self.tracking_mode == "hand":
                # El tespiti
                hands = self.detect_hand(frame, gray)
                
                if hands:
                    # En büyük eli al
                    x, y, w, h, confidence = max(hands, key=lambda h: h[4])
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    self.motion_history.append((center_x, center_y))
                    
                    # Kıyafet rengini tespit et
                    clothing_color = self.detect_clothing_color(frame, x, y, w, h)
                    
                    track_info = {
                        'bbox': (x, y, w, h),
                        'center': (center_x, center_y),
                        'area': w * h,
                        'type': 'el',
                        'confidence': confidence,
                        'clothing_color': clothing_color
                    }
                    
                    # El şeklini çiz
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"EL ({confidence:.2f})", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    if clothing_color != "Belirlenemedi":
                        cv2.putText(frame, f"Kiyafet: {clothing_color}", (x, y+h+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Takip bilgilerini çiz
            if track_info:
                frame = self.draw_tracking_info(frame, track_info)
            
            # Renk paletini çiz
            frame = self.draw_color_palette(frame)
            
            # Mod bilgisi
            cv2.putText(frame, f"MOD: {self.tracking_mode.upper()}", 
                       (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # FPS bilgisi
            cv2.putText(frame, "FPS: Cam", (frame.shape[1] - 150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Pencereyi göster
            cv2.imshow('Gelişmiş Nesne Takip Sistemi', frame)
            
            # Maske penceresi
            if track_info and 'mask' in track_info:
                cv2.imshow('Maske Görünümü', track_info['mask'])
            
            # Tuş kontrolleri
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Mod değiştir
                modes = ["color", "motion", "hand"]
                current_idx = modes.index(self.tracking_mode)
                self.tracking_mode = modes[(current_idx + 1) % len(modes)]
                self.motion_history.clear()
                print(f"Mod değiştirildi: {self.tracking_mode}")
            elif key == ord('i'):
                self.show_info = not self.show_info
            elif key == ord('s'):
                # Ekran görüntüsü al
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, original_frame)
                print(f"Ekran görüntüsü kaydedildi: {filename}")
            elif ord('1') <= key <= ord('6'):
                # Renk seçimi
                color_keys = list(self.color_ranges.keys())
                color_index = key - ord('1')
                if color_index < len(color_keys):
                    self.tracked_color = color_keys[color_index]
                    print(f"Takip edilen renk: {self.tracked_color}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
    def analyze_colors(self, image_path):
        """Görüntüdeki renkleri analiz et"""
        image = cv2.imread(image_path)
        if image is None:
            print("Görüntü yüklenemedi!")
            return
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        plt.figure(figsize=(15, 10))
        
        # Orijinal
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Orijinal Görüntü')
        plt.axis('off')
        
        # HSV kanalları
        plt.subplot(2, 3, 2)
        plt.imshow(hsv[:,:,0], cmap='hsv')
        plt.title('Hue (Renk Tonu)')
        plt.colorbar()
        
        plt.subplot(2, 3, 3)
        plt.imshow(hsv[:,:,1], cmap='gray')
        plt.title('Saturation (Doygunluk)')
        plt.colorbar()
        
        plt.subplot(2, 3, 4)
        plt.imshow(hsv[:,:,2], cmap='gray')
        plt.title('Value (Parlaklık)')
        plt.colorbar()
        
        # Renk maskeleri
        mask_display = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        
        colors_found = []
        for i, (color_name, ranges) in enumerate(self.color_ranges.items()):
            mask = None
            for lower, upper in ranges:
                lower_np = np.array(lower)
                upper_np = np.array(upper)
                color_mask = cv2.inRange(hsv, lower_np, upper_np)
                
                if mask is None:
                    mask = color_mask
                else:
                    mask = cv2.bitwise_or(mask, color_mask)
            
            if mask is not None:
                # Renk ataması
                if color_name == "Kirmizi":
                    display_color = (0, 0, 255)
                elif color_name == "Yesil":
                    display_color = (0, 255, 0)
                elif color_name == "Mavi":
                    display_color = (255, 0, 0)
                elif color_name == "Sari":
                    display_color = (0, 255, 255)
                elif color_name == "Turuncu":
                    display_color = (0, 165, 255)
                elif color_name == "Pembe":
                    display_color = (255, 0, 255)
                else:
                    display_color = (255, 255, 255)
                
                mask_display[mask > 0] = display_color
                
                # Renk oranı
                color_ratio = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
                if color_ratio > 0.01:  # %1'den fazla ise
                    colors_found.append((color_name, color_ratio))
        
        plt.subplot(2, 3, 5)
        plt.imshow(mask_display)
        plt.title('Tespit Edilen Renkler')
        plt.axis('off')
        
        # Renk dağılımı
        plt.subplot(2, 3, 6)
        if colors_found:
            colors, ratios = zip(*colors_found)
            plt.bar(colors, ratios, color=['red', 'green', 'blue', 'yellow', 'orange', 'pink'])
            plt.title('Renk Dağılımı')
            plt.ylabel('Oran')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("\nRENK ANALİZİ SONUÇLARI:")
        for color_name, ratio in colors_found:
            print(f"  {color_name}: %{ratio*100:.1f}")

# Basit ve çalışır versiyon
def simple_color_tracker():
    """Basit renk takip sistemi"""
    cap = cv2.VideoCapture(0)
    
    # Renk aralıkları
    color_ranges = {
        "mavi": ([100, 150, 0], [140, 255, 255]),
        "kirmizi": ([0, 150, 50], [10, 255, 255]),
        "yesil": ([40, 50, 50], [80, 255, 255]),
        "sari": ([20, 100, 100], [30, 255, 255])
    }
    
    current_color = "mavi"
    
    print("BASİT RENK TAKİP SİSTEMİ")
    print("SPACE: Renk değiştir")
    print("1: Mavi, 2: Kırmızı, 3: Yeşil, 4: Sarı")
    print("q: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Seçili renk için maske
        lower, upper = color_ranges[current_color]
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        
        mask = cv2.inRange(hsv, lower_np, upper_np)
        
        # Gürültü temizleme
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # En büyük kontur
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                x, y, w, h = cv2.boundingRect(largest)
                
                # Çerçeve çiz
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Merkez
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Bilgi
                cv2.putText(frame, f"RENK: {current_color.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"X:{center_x}, Y:{center_y}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Pencere
        cv2.imshow('Basit Renk Takip', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Renk değiştir
            colors = list(color_ranges.keys())
            current_idx = colors.index(current_color)
            current_color = colors[(current_idx + 1) % len(colors)]
        elif ord('1') <= key <= ord('4'):
            idx = key - ord('1')
            colors = list(color_ranges.keys())
            if idx < len(colors):
                current_color = colors[idx]
    
    cap.release()
    cv2.destroyAllWindows()

def hand_tracker_simple():
    """Basit el takip sistemi"""
    cap = cv2.VideoCapture(0)
    
    print("BASİT EL TAKİP SİSTEMİ")
    print("Deri rengini takip eder")
    print("q: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Deri rengi aralığı
        lower_skin = np.array([0, 30, 60])
        upper_skin = np.array([20, 150, 255])
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Gürültü temizleme
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Konturlar
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # En büyük kontur (el)
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area > 2000:  # El boyutu
                x, y, w, h = cv2.boundingRect(largest)
                
                # El çerçevesi
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "EL", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Kıyafet rengi tespiti (alt bölge)
                clothing_roi = frame[y+h//2:y+h, x:x+w]
                if clothing_roi.size > 0:
                    hsv_clothing = cv2.cvtColor(clothing_roi, cv2.COLOR_BGR2HSV)
                    
                    # Renk histogramı
                    hist_h = cv2.calcHist([hsv_clothing], [0], None, [180], [0, 180])
                    dominant_hue = np.argmax(hist_h)
                    
                    # Renk ismi
                    color_name = "Belirlenemedi"
                    if 0 <= dominant_hue < 10 or 170 <= dominant_hue < 180:
                        color_name = "Kirmizi"
                    elif 10 <= dominant_hue < 25:
                        color_name = "Turuncu"
                    elif 25 <= dominant_hue < 35:
                        color_name = "Sari"
                    elif 35 <= dominant_hue < 85:
                        color_name = "Yesil"
                    elif 85 <= dominant_hue < 130:
                        color_name = "Mavi"
                    elif 130 <= dominant_hue < 170:
                        color_name = "Pembe"
                    
                    cv2.putText(frame, f"Kiyafet: {color_name}", (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Görüntüleri göster
        cv2.imshow('El Takip', frame)
        cv2.imshow('Deri Maskesi', mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Ana program
if __name__ == "__main__":
    print("NESNE TAKİP SİSTEMİ")
    print("="*40)
    print("1. Gelişmiş Takip Sistemi (3 mod)")
    print("2. Basit Renk Takip Sistemi")
    print("3. Basit El Takip Sistemi")
    print("4. Renk Analizi (görüntüden)")
    
    choice = input("\nSeçiminiz (1-4): ")
    
    if choice == "1":
        tracker = AdvancedObjectTracker()
        tracker.run()
    elif choice == "2":
        simple_color_tracker()
    elif choice == "3":
        hand_tracker_simple()
    elif choice == "4":
        image_path = input("Analiz edilecek görüntü dosyası: ")
        tracker = AdvancedObjectTracker()
        tracker.analyze_colors(image_path)
    else:
        print("Geçersiz seçim!")