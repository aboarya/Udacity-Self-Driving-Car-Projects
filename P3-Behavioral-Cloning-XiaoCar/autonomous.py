from driver.piborg import Driver
from model import save_data

def main():
    driver = Driver(autonomous=True)
    try:
        print()
        print('Press CTRL+C to quit')
        driver.drive()        
    except SystemExit:
        print("exited")
    except KeyboardInterrupt:
        # CTRL+C exit, disable all drives
        save_data(driver.DATA, file_path='robot-predicted.p')
        driver.PBR.MotorsOff()

if __name__ == '__main__':
    main()