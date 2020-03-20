from gui.widgets_configurations.anchor_left import ANCHOR_LEFT
from gui.widgets_configurations.button import BUTTON_CONFIG
from gui.widgets_configurations.copyright import COPYRIGHT_CONFIG
from gui.widgets_configurations.logo import LOGO_CONFIG_INIT, LOGO_CONFIG_ADVANCED
from gui.widgets_configurations.menu_button import MENU_BUTTON_CONFIG


def set_button_configuration(btn, text):
    btn.configure(BUTTON_CONFIG)
    btn.configure(text=text)


def set_logo_configuration(logo, image):
    logo.configure(LOGO_CONFIG_INIT)
    logo.configure(image=image)
    logo.configure(LOGO_CONFIG_ADVANCED)


def set_copyright_configuration(copy_right):
    copy_right.configure(COPYRIGHT_CONFIG)


def set_menu_button_configuration(menubutton):
    menubutton.configure(MENU_BUTTON_CONFIG)


def set_widget_to_left(widget):
    widget.configure(ANCHOR_LEFT)
