#!/usr/bin/env python3
import PySimpleGUI as sg

def getSceneNumber():

    input_layout = [
        [sg.Text("Enter a number between 1 and 10:")],
        [sg.InputText(key='-NUMBER-')],
        [sg.Button("OK"), sg.Button("Cancel")]
    ]
    input_window = sg.Window("Enter Number", input_layout)

    while True:
        input_event, input_values = input_window.read()
        if input_event == sg.WINDOW_CLOSED or input_event == "Cancel":
            break
        elif input_event == "OK":
            # Get the value entered by the user
            scene_number = input_values['-NUMBER-']
            try:
                scene_number = int(scene_number)
                if 1 <= scene_number <= 10:
                    sg.popup(f"You entered: {scene_number}")
                    break
                else:
                    sg.popup("Chose between scene 1 and 10.")
            except ValueError:
                sg.popup("Please enter a valid number.")

    input_window.close()

    return scene_number

def main():
    # Define the layout of the GUI
    layout = [
        [sg.Text('', size=(20,1))],  # Empty text element for spacing
        [sg.Button("Play in Real Life", size=(15, 2), button_color=('white', 'black')), 
         sg.Button("Play from Scene", size=(15, 2), button_color=('white', 'black'))],
        [sg.Text('', size=(20,1))]  # Empty text element for spacing
    ]

    window_size = (500,400)
    # Create the GUI window
    window = sg.Window("SAVI TP2", layout, finalize=True, size=window_size)  

    # Event loop to process events and interact with the GUI
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == "Play in Real Life":
            sg.popup("Play in Real Life functionality clicked!")
        elif event == "Play from Scene":
            # Open a pop-up window with an input field for the user to enter a number between 1 and 10
            scene_number = getSceneNumber()
    # Close the main window when the loop exits
    window.close()

if __name__ == "__main__":
    main()
