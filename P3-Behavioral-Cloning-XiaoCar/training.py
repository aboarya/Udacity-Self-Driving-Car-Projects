from driver.piborg import Driver

def main():
    driver = Driver()
    try:
        print()
        print('Press CTRL+C to quit')
        driver.drive()        
    except SystemExit:
        print("exited")
    except KeyboardInterrupt:
        # CTRL+C exit, disable all drives
        driver.PBR.MotorsOff()

if __name__ == '__main__':
    main()