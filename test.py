import pyautogui

# Move the mouse to the center of the screen
screen_width, screen_height = pyautogui.size()
print(screen_width, screen_height)
center_x, center_y = screen_width / 2, screen_height / 2
pyautogui.moveTo(center_x, center_y, duration=1) 