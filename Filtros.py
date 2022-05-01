import numpy as np
import cv2
import time
import mediapipe as mp
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def find_hands(frame, hand_detector, show=True, halve=True):

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    frame.flags.writeable = False
    results = hand_detector.process(frame)
    frame.flags.writeable = True

    hands_coord = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Mostrar esqueleto de las manos
            if show:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # si halve=True, el tamaño del rectángulo se divide por dos
            if halve:
                w = int(abs(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x - hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x) * frame.shape[1]/2)
                h = int(abs(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y) * frame.shape[1]/2)
                x = int(min(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x) * frame.shape[1]+ w/2)
                y = int(min(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)* frame.shape[0] + h/2)
            else:
                x = int(min(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x) * frame.shape[1])
                y = int(min(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)* frame.shape[0])
                w = int(abs(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x - hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x) * frame.shape[1])
                h = int(abs(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y) * frame.shape[1])

            hands_coord.append((x,y,w,h))

    return hands_coord

def apply_shift(in_path, out_path, show_boxes = True, method='meanshift', start_auto=False):

    # Máscara para la galleta
    template = cv2.imread('./images/cookie80.png')
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    template_mask = np.zeros(shape=template.shape)
    template_mask[:,:,0] = template_gray < 250
    template_mask[:,:,1] = template_gray < 250
    template_mask[:,:,2] = template_gray < 250
    cap = cv2.VideoCapture(in_path) # fichero a procesar

    if (cap.isOpened()):

        # leemos el primer frame
        ret, frame = cap.read()
        
        # Flip frame 
        frame = cv2.flip(frame, 1)

        # creamos el video de salida
        if out_path != None:
            height, width, layers = frame.shape
            size = (width, height)
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        # Inicialización del tiempo
        pTime = 0

        # Inicialización del detector de manos
        hand_detector = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Inicialización MeanShift
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        track_window = None
        roi_hist = None
        init = True

        # mientras se haya podido leer el siguiente frame
        while(cap.isOpened() and ret):

            # Salir
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q') or k == 27:
                break

            # Inicialización
            if init == True:

                # Buscar las manos
                frame2 = frame.copy()
                hands = find_hands(frame, hand_detector, show=show_boxes, halve=True)

                if len(hands) > 0:

                    # Encontrar coordenadas y tamaño de la mano
                    track_window = hands[0]
                    (x,y,w,h) = track_window

                    # set up the ROI for tracking
                    roi = frame2[y:y+h, x:x+w]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)),
                    np.array((180.,255.,255.)))
                    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                    
                    # Draw it on image
                    cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                    
                    if k == ord(' ') or start_auto:
                        init = False

            else:
                
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                
                # apply meanshift to get the new location
                ret, track_window = cv2.meanShift(dst, track_window, term_crit) if method == 'meanshift' else cv2.CamShift(dst, track_window, term_crit)
                x, y, w, h = track_window
                
                # Poner galleta
                x1 = np.clip(int(x+w/2), int(template.shape[1]/2), frame.shape[1] - int(template.shape[1]/2))
                y1 = np.clip(int(y+h/2), int(template.shape[0]/2), frame.shape[0] - int(template.shape[0]/2))
                np.putmask(frame[int(y1-template.shape[0]/2):int(y1+template.shape[0]/2), int(x1-template.shape[1]/2):int(x1+template.shape[1]/2), :], template_mask, template)
                
                # Draw it on image
                if show_boxes:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                

            # FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, f'FPS:{int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Mostrar frame
            cv2.imshow('frame', frame)

            # Guardar frame
            if out_path is not None:
                out.write(frame)

            # leer siguiente frame
            ret, frame = cap.read()

            # Flip frame 
            frame = cv2.flip(frame, 1)


        if out_path != None:
            out.release()

    cap.release()
    cv2.destroyAllWindows()

def apply_template_matching(in_path, out_path, show_boxes = True, start_auto=False):

    # Máscara para la galleta
    template = cv2.imread('./images/cookie80.png')
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    template_mask = np.zeros(shape=template.shape)
    template_mask[:,:,0] = template_gray < 250
    template_mask[:,:,1] = template_gray < 250
    template_mask[:,:,2] = template_gray < 250

    cap = cv2.VideoCapture(in_path) # fichero a procesar o webcam

    mode = True # Mostrar frame o imagen de distancias

    if (cap.isOpened()):

        # leemos el primer frame
        ret, frame = cap.read()
        
        # Flip frame 
        frame = cv2.flip(frame, 1)

        # creamos el video de salida
        if out_path != None:
            height, width, layers = frame.shape
            size = (width, height)
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        # Inicialización del tiempo
        pTime = 0

        # Inicialización del detector de manos
        hand_detector = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Inicialización TemplateMatching
        init = True

        # mientras se haya podido leer el siguiente frame
        while(cap.isOpened() and ret):

            # Salir
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q') or k == 27:
                break

            if k == ord('p'):
                mode = not mode

            # Inicialización
            if init == True:

                # Buscar las manos
                frame2 = frame.copy()
                hands = find_hands(frame, hand_detector, show=show_boxes, halve=True)

                if len(hands) > 0:

                    # Encontrar coordenadas y tamaño de la mano
                    track_window = hands[0]
                    (x,y,w,h) = track_window

                    # set up the ROI for tracking
                    roi = frame2[y:y+h, x:x+w]
                    
                    # Draw it on image
                    cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                    
                    if k == ord(' ') or start_auto:
                        init = False
                        res = frame

            else:
                
                # Apply template Matching
                res = cv2.matchTemplate(frame ,roi, cv2.TM_SQDIFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                (x, y) = min_loc # Usar min_loc o max_loc según el método usado para el cálculo de distancias
                
                # Poner galleta
                x1 = np.clip(int(x+w/2), int(template.shape[1]/2), frame.shape[1] - int(template.shape[1]/2))
                y1 = np.clip(int(y+h/2), int(template.shape[0]/2), frame.shape[0] - int(template.shape[0]/2))
                np.putmask(frame[int(y1-template.shape[0]/2):int(y1+template.shape[0]/2), int(x1-template.shape[1]/2):int(x1+template.shape[1]/2), :], template_mask, template)
                
                # Draw it on image
                if show_boxes:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
                

            # FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, f'FPS:{int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Mostrar frame
            
            if init:
                cv2.imshow('frame', frame)
            else:
                if mode:
                    cv2.imshow('frame', frame)
                else:
                    cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX)
                    cv2.imshow('frame', res)

            # Guardar frame
            if out_path is not None:
                out.write(frame)

            # leer siguiente frame
            ret, frame = cap.read()

            # Flip frame 
            frame = cv2.flip(frame, 1)


        if out_path != None:
            out.release()

    cap.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Apply Kalman or Particle filter.')
parser.add_argument('--filter', dest='filter', type=str, default='meanshift',
                    help='Chosen filter.')
parser.add_argument('--out_path', dest='out_path', type=str, default=None,
                    help='Out path of video.')
parser.add_argument('--show_boxes', dest='show_boxes', action='store_true', default=False,
                    help='Either show boxes of the detections along with hand skeletons or not.')
parser.add_argument('--start_auto', dest='start_auto', action='store_true', default=False,
                    help='Either capture automatically hand in first frame or wait until space bar is pressed.')

args = parser.parse_args()

if args.filter == 'meanshift' or args.filter == 'camshift':
    apply_shift(in_path=0, out_path=args.out_path, show_boxes=args.show_boxes, method=args.filter, start_auto=args.start_auto)
else:
    apply_template_matching(in_path=0, out_path=args.out_path, show_boxes=args.show_boxes, start_auto=args.start_auto)