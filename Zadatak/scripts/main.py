import imageProcessing as imageProcessing
import videoProcessing as videoProcessing

if __name__ == '__main__':
    
    choice = input("Enter 'i' for image processing or 'v' for video processing: ")
    if choice == 'i':
        imageProcessing.testImages()
    elif choice == 'v':
        videoProcessing.testVideo()
    else:
        print("Invalid choice. Please enter 'i' for image processing or 'v' for video processing.")
    