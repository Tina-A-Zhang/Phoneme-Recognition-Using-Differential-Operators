import time
import sys

def countdown(time_sec):
    while time_sec:
        mins, secs = divmod(time_sec, 60)
        hours, mins = divmod(mins, 60)
        timeformat = '{:02d}:{:02d}:{:02d}'.format(hours, mins, secs)
        print('Job starts in: ', timeformat, end='\r')
        time.sleep(1)
        time_sec -= 1

def main(argv):
    countdown(int(sys.argv[1]))

if __name__ == "__main__":
    main(int(sys.argv[1]))