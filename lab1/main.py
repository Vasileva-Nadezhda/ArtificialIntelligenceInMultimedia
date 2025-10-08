import cv2, numpy as np

# Любезно предоставленная функция для поиска центроида
def find_centroid(img):
  contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0
    return (cx, cy)
  return None


def main():
   
   # Готовимся
   video_path = './stardew.mp4'
   cap = cv2.VideoCapture(video_path)
   writer = cv2.VideoWriter('tracked_stardew.mp4', 
                            cv2.VideoWriter_fourcc(*'MP4V'), 
                            25, (int(cap.get(3)),int(cap.get(4))))
   ret = True
   text = '00.00'
   org = (150, 40)
   font = cv2.FONT_HERSHEY_SIMPLEX
   font_scale = 1
   color = (0, 140, 240)
   coordinates = []

   # Непосредственно обработка видео
   while ret:
      
      ret, frame_bgr = cap.read()
      if ret:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        n_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Раз в три кадра обновляем таймер и ставим новую метку
        if n_frame % 3 == 0:
            msec = n_frame / (cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) * 1000))
            text = f"{int(msec/1000):02d}.{(int(msec)%1000):02d}"
            image_thresholded = cv2.inRange(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), (90, 75, 75), (150, 200, 200))
            coordinates.append(find_centroid(image_thresholded))

        for c in coordinates:
                cv2.drawMarker(frame_rgb, c, color=(255, 255, 255), thickness=1,
                        markerType=cv2.MARKER_STAR, line_type=cv2.LINE_AA, markerSize=15)
        cv2.putText(frame_rgb, text, org, font, font_scale, color)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        writer.write(frame_bgr)

   # Сохраняем
   writer.release()


if __name__ == "__main__":
   main()