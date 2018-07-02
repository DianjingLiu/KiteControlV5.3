# -*- coding:utf-8 -*-  
import os
import pygame
# from pygame.locals import *

AlignLeft = 0x01
AlignRight = 0x02
AlignHorizonCenter = 0x03
AlignTop = 0x10
AlignBottom = 0x20
AlignVerticalCenter = 0x30

x = 50
y = 50
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)
pygame.init()
pygame.display.set_caption("View demo")
screen = pygame.display.set_mode((800, 600), 0, 32)

# font1 = pygame.font.Font(None, 72)
# font2 = pygame.font.Font(None, 48)
font_text = pygame.font.Font(None, 24)

white = 255, 255, 255
red = 255, 0, 0
green = 0, 255, 0
blue = 0, 0, 255
yellow = 255, 255, 0
orange = 255, 129, 66


def print_all(RevolutionsL, RevolutionsR, RotateSpeedL, RotateSpeedR, TensorVL, TensorVR):
    # 左右靠左5，上下50居中
    print_text("LeftTensor", font_text, 0 + 5, 100 / 2, AlignLeft | AlignVerticalCenter, red)
    print_progress(100 + 5, 100 / 2, 250, 40, AlignLeft | AlignVerticalCenter, TensorVL / 3.3, InAlign=AlignLeft,
                   color=red)
    print_text(str('%.3f' % TensorVL), font_text, 100 + 5 + 250 - 50, 100 / 2, AlignLeft | AlignVerticalCenter, white)

    print_text("RightTensor", font_text, 800 - 5, 100 / 2, AlignRight | AlignVerticalCenter, red)
    print_progress(800 - 100 - 5, 100 / 2, 250, 40, AlignRight | AlignVerticalCenter, TensorVR / 3.3,
                   InAlign=AlignRight, color=red)
    print_text(str('%.3f' % TensorVR), font_text, 800 - 100 - 5 - 250 + 50, 100 / 2, AlignRight | AlignVerticalCenter,
               white)

    # 左右100居中，上下750居中
    print_text("LeftSpeed", font_text, 75, 560, AlignHorizonCenter | AlignVerticalCenter, blue)
    print_progress(75, 530, 40, 400, AlignHorizonCenter | AlignBottom, RotateSpeedL / 4 + 0.5, InAlign=AlignBottom,
                   color=blue)
    print_text(str('%.4f' % RotateSpeedL), font_text, 75, 120, AlignHorizonCenter | AlignVerticalCenter, white)

    print_text("LeftLength", font_text, 200, 560, AlignHorizonCenter | AlignVerticalCenter, green)
    print_progress(200, 530, 50, 400, AlignHorizonCenter | AlignBottom, RevolutionsL / 200 + 0.5, InAlign=AlignBottom,
                   color=green)
    print_text(str('%.4f' % RevolutionsL), font_text, 200, 120, AlignHorizonCenter | AlignVerticalCenter, white)

    print_text("RightLength", font_text, 600, 560, AlignHorizonCenter | AlignVerticalCenter, green)
    print_progress(600, 530, 50, 400, AlignHorizonCenter | AlignBottom, RevolutionsR / 200 + 0.5, InAlign=AlignBottom,
                   color=green)
    print_text(str('%.4f' % RevolutionsR), font_text, 600, 120, AlignHorizonCenter | AlignVerticalCenter, white)

    print_text("RightSpeed", font_text, 650 + 75, 560, AlignHorizonCenter | AlignVerticalCenter, blue)
    print_progress(650 + 75, 530, 40, 400, AlignHorizonCenter | AlignBottom, RotateSpeedR / 4 + 0.5,
                   InAlign=AlignBottom, color=blue)
    print_text(str('%.4f' % RotateSpeedR), font_text, 650 + 75, 120, AlignHorizonCenter | AlignVerticalCenter, white)



def print_Operate(Rotate_button):
    print_text("Different", font_text, 400, 300 - 30, AlignHorizonCenter | AlignVerticalCenter, green)
    print_progress(400, 300, 300, 40, AlignHorizonCenter | AlignVerticalCenter, (Rotate_button) / 2 + 0.5,
                   InAlign=AlignHorizonCenter, color=blue)
    print_text(str('%.4f' % (Rotate_button)), font_text, 400, 300 + 30,
               AlignHorizonCenter | AlignVerticalCenter, white)


def print_refresh(refreshcount):
    print_text("Refresh", font_text, 400, 300 + 100, AlignHorizonCenter | AlignVerticalCenter, green)
    print_text(str('%.0f' % refreshcount), font_text, 400, 300 + 120, AlignHorizonCenter | AlignVerticalCenter, white)

def print_track_time(track_time, ok):
    print_text("Track time (ms)", font_text, 400, 300 + 220, AlignHorizonCenter | AlignVerticalCenter, green)
    print_text(str('%.0f' % track_time), font_text, 400, 300 + 240, AlignHorizonCenter | AlignVerticalCenter, white if ok else red)

def print_mode(MODE, AI_MODEL):
    print_text("Control Mode", font_text, 400, 300 + 160, AlignHorizonCenter | AlignVerticalCenter, green)
    if MODE == 1:
        print_text("Manual", font_text, 400, 300 + 180, AlignHorizonCenter | AlignVerticalCenter, white)
    elif MODE == 2:
        print_text("Auto {:d}".format(AI_MODEL), font_text, 400, 300 + 180, AlignHorizonCenter | AlignVerticalCenter, white)

def print_tipmessage(tip="", step=0):
    print_text(tip, font_text, 400, 300 + 100, AlignHorizonCenter | AlignVerticalCenter, green)
    print_text(str('%.0f' % step), font_text, 400, 300 + 120, AlignHorizonCenter | AlignVerticalCenter, white)


# Center text out
def print_text(text, font, Colx, Coly, Align=AlignLeft | AlignTop, color=(255, 255, 255)):
    imgText = font.render(text, True, color)
    strwidth, strheight = font.size(text)
    AlignX = Colx
    AlignY = Coly
    if (Align & 0x0F) == AlignLeft:
        AlignX = Colx
    elif (Align & 0x0F) == AlignRight:
        AlignX = Colx - strwidth
    elif (Align & 0x0F) == AlignHorizonCenter:
        AlignX = Colx - strwidth / 2

    if (Align & 0xF0) == AlignTop:
        AlignY = Coly
    elif (Align & 0xF0) == AlignBottom:
        AlignY = Coly - strheight
    elif (Align & 0xF0) == AlignVerticalCenter:
        AlignY = Coly - strheight / 2

    screen.blit(imgText, (AlignX, AlignY))


def print_progress(Colx, Coly, width, height, OutAlign=AlignLeft | AlignTop, Progress=0.0, InAlign=AlignLeft,
                   color=(255, 255, 255)):
    AlignX = Colx
    AlignY = Coly
    if (OutAlign & 0x0F) == AlignLeft:
        AlignX = Colx
    elif (OutAlign & 0x0F) == AlignRight:
        AlignX = Colx - width
    elif (OutAlign & 0x0F) == AlignHorizonCenter:
        AlignX = Colx - width / 2

    if (OutAlign & 0xF0) == AlignTop:
        AlignY = Coly
    elif (OutAlign & 0xF0) == AlignBottom:
        AlignY = Coly - height
    elif (OutAlign & 0xF0) == AlignVerticalCenter:
        AlignY = Coly - height / 2
    pygame.draw.rect(screen, (255, 255, 255), [AlignX, AlignY, width, height], 2)

    if Progress < 0:
        Progress = 0
    elif Progress > 1:
        Progress = 1
    InsideX = AlignX + 2
    InsideY = AlignY + 2
    InsideWidth = width - 3
    InsideHeight = height - 3
    if (InAlign & 0x0F) == AlignLeft:
        InsideX = AlignX + 2
        InsideWidth = Progress * width - 3
    elif (InAlign & 0x0F) == AlignRight:
        InsideX = AlignX + width - Progress * width + 2
        InsideWidth = Progress * width - 3
    elif (InAlign & 0x0F) == AlignHorizonCenter:
        InsideX = AlignX + width / 2 - Progress * width / 2 + 2
        InsideWidth = Progress * width - 3

    if (InAlign & 0xF0) == AlignTop:
        InsideY = AlignY + 2
        InsideHeight = Progress * height - 3
    elif (InAlign & 0xF0) == AlignBottom:
        InsideY = AlignY + height - Progress * height + 2
        InsideHeight = Progress * height - 3
    elif (InAlign & 0xF0) == AlignVerticalCenter:
        InsideY = AlignY + height / 2 - Progress * height / 2 + 2
        InsideHeight = Progress * height - 3
    pygame.draw.rect(screen, color, [InsideX, InsideY, InsideWidth, InsideHeight], 0)


def print_close():
    pygame.quit()
