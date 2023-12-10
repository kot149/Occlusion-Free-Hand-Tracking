import cv2
import time
from concurrent.futures import ThreadPoolExecutor
import traceback

class captureClass():
    def __init__(self, cap_number):
        self.cap = cv2.VideoCapture(cap_number, cv2.CAP_DSHOW)

    def readFrame(self):
        ret, self.frame = self.cap.read()
        return ret

    def getFrame(self):
        return self.frame

    def capRelease(self):
        self.cap.release()

def threadCapture(cap_number1, cap_number2):
    cap_obj1 = captureClass(cap_number1)
    cap_obj2 = captureClass(cap_number2)
    count = 0
    while True:
        cap_obj1.readFrame()
        cap_obj2.readFrame()
        frame1 = cap_obj1.getFrame()
        frame2 = cap_obj2.getFrame()
        frame1 = cv2.resize(frame1, (320,240))
        frame2 = cv2.resize(frame2, (320,240))

        frame = cv2.vconcat([frame1, frame2])
        frame = cv2.putText(frame, "count: "+str(count), (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),1,cv2.LINE_8)

        cv2.imshow("frame", frame)
        count += 1

        lastkey = cv2.waitKey(1)
        if lastkey == ord("q"):
            cap_obj1.release()
            cap_obj2.release()
            cv2.destroyAllWindows()
            break
        if lastkey == ord("c"):
            cv2.imwrite("frame.png", frame)


if __name__ == "__main__":

    cap_number1 = 0
    cap_number2 = 0

    executor = ThreadPoolExecutor(max_workers=3)
    camera_future = executor.submit(threadCapture, cap_number1, cap_number2)

    while True:
        if camera_future.running() == False:
            print("Camera shutdown")
            executor.shutdown()
            break
        else:
            time.sleep(5)
            print("Sleeping 5 seconds ...")

    print("program complete")