#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on Jänner 09, 2026, at 11:15
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2025.1.1')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = '1.083Hz_65bcpm_FPB_circle'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'session_admin': 'xxx',
    'UPN-ID': '999',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['UPN-ID'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\weberbe\\Nextcloud\\__PsychoPy\\__Paradigmen_HD-Labor__\\fast_paced_breathing\\1.083Hz_65bcpm_FPB_circle_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('error')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='HD-Lab', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('FPB_ins_key') is None:
        # initialise FPB_ins_key
        FPB_ins_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='FPB_ins_key',
        )
    if deviceManager.getDevice('thx_key') is None:
        # initialise thx_key
        thx_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='thx_key',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "_settings" ---
    # Run 'Begin Experiment' code from set_things
    ## ########################################################################## ##
    ## ##                                                                      ## ##
    ## ##              FAST PACED BREATHING (FPB)/HYPERVENTILATION             ## ##
    ## ##                                                                      ## ##
    ## #########################|12/2025|bw (bernhard.weber@uni-graz.at)|CC-BY|## ##
    print(data.getDateStr(format="%H:%M:%S.%m"))   #local: set/mark start of current run
    
    ##  enable parallel port for markers  ##
    from psychopy import parallel
    Marker = parallel.ParallelPort(address=0x3FF8)
    
    ##  Options for DEBUG/RESARCH-mode  ##
    DEBUG = 0   #0=research-mode; 1=debug-mode 
    if DEBUG:
        BREATHING_DUR = 10          #debug: breathing duration in [s]
    else:
        BREATHING_DUR = 120         #breathing duration in [s]
    
    ## breathing frequency
    BCPM = 65                       #65 breathing-cycles per minute = 1.083Hz
    
    
    ##  Options for ONSCREEN-KEYBOARD (touchscreen WACOM ONE)  ##
    ONSCREENKEYBOARD = 0   #0=NO onscreen keyboard; 1=ONSCREEN-KEYBOARD with touchscreen WACOM ONE
    if ONSCREENKEYBOARD:
        header_pos_y = .40
        main_pos_y   = .225
        main_pos_y_px= 270
        btn_pos_y    = .05
        hint_pos_y   = 0
    else:
        header_pos_y = .40
        main_pos_y   = 0
        main_pos_y_px= 0
        btn_pos_y    = -.25
        hint_pos_y   = -.45
    
    
    ##  initialize some vars  ##
    win.mouseVisible = True
    
    
    # --- Initialize components for Routine "FPB_ins" ---
    FPB_ins_header = visual.TextStim(win=win, name='FPB_ins_header',
        text='(schnelle) Pendelatmung',
        font='Arial',
        pos=(0, .4), draggable=False, height=0.05, wrapWidth=1.0, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    FPB_ins_txt = visual.TextStim(win=win, name='FPB_ins_txt',
        text='Es folgt die schnelle Atembedingung. \nAtme nun bitte in der mit dem Atemkreis vorgegebenen Frequenz.\n\nWenn du bereit bist > "Weiter"',
        font='Arial',
        pos=(0, main_pos_y), draggable=False, height=0.0325, wrapWidth=1.25, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    FPB_ins_key = keyboard.Keyboard(deviceName='FPB_ins_key')
    FPB_ins_btn = visual.Rect(
        win=win, name='FPB_ins_btn',
        width=(0.15, 0.05)[0], height=(0.15, 0.05)[1],
        ori=0.0, pos=(0, btn_pos_y), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='lightgrey',
        opacity=None, depth=-4.0, interpolate=True)
    FPB_ins_hint = visual.TextStim(win=win, name='FPB_ins_hint',
        text='Weiter',
        font='Open Sans',
        pos=(0, btn_pos_y), draggable=False, height=0.025, wrapWidth=None, ori=0.0, 
        color='darkgreen', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    FPB_ins_mse = event.Mouse(win=win)
    x, y = [None, None]
    FPB_ins_mse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "go4FPB" ---
    go4it_txt = visual.TextStim(win=win, name='go4it_txt',
        text='...es beginnen die schnellen Atemzüge...',
        font='Open Sans',
        pos=(0, main_pos_y), draggable=False, height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "FPB" ---
    # Run 'Begin Experiment' code from FPB_code
    ###   ===== CONFIGURATION =====   ###
    # BCPM ... breathing-cycles per minute (> for Hyperventilation/fast breathing)
    # -----------------------------------
    #BCPM = 6                   # 6 breathing-cycles per minute = 0.100Hz > STANDARD SLOW PACED BREATHING
    #BCPM = 36                  #36 breathing-cycles per minute = 0.600Hz > STANDARD FAST PACED BREATHING
    #BCPM = 50                  #50 breathing-cycles per minute = 0.833Hz - fast paced HV (Angerbauer 2025)
    #BCPM = 60                  #60 breathing-cycles per minute = 1.000Hz
    #BCPM = 65                  #65 breathing-cycles per minute = 1.083Hz
    #BCPM = 70                  #70 breathing-cycles per minute = 1.166Hz
    #BCPM = 75                  #70 breathing-cycles per minute = 1.250Hz
    
    
    ##  duration of (fast paced) breathing task  ##
    # >>(global) '_settings'- routine >> BREATHING_DUR = 10.0        #breathing duration in [s]
    
    #CYCLE_TIME = 10.0          # Gesamtdauer eines Atemzyklus in Sekunden
    CYCLE_TIME = 60/BCPM        # 60s/breathing-cycles_per_minute
    
    #INHALE_RATIO = 0.60        # ratio of inhale (60% = 6s for a 10s cycle)
    INHALE_RATIO = 0.50         # 50% inhale - 50% exhale
    PAUSE_TIME = 0.00           # pause during in- and exhale
    MIN_SIZE_PERCENT = 0.275    # 'residual volume' (xx% of maximum)
    #MAX_RADIUS = 300           # max. radius [px]
    #MAX_RADIUS = 0.3           # max. radius [norm] units; (0.3 = 30% der Fensterhöhe)
    MAX_DIAMETER = 0.6          # max. diameter [norm] units (0.7 = 70% of screen/windowheight)
    
    ##  define some colors (RGB from -1 to 1)
    #COLOR_INHALE = [0.0, 0.5, 1.0]             #blue
    COLOR_INHALE = [1.0000, 0.4275, 0.5137]     #breathing circle color (inhale JMT neu)
    COL_INH_LINE = [1.0000, 0.2000, 0.6000]     #breathing cirlce line color
    #COLOR_EXHALE = [1.0000, 0.3000, 0.5000]    #pink
    COLOR_EXHALE = [0.0588, 0.6157, 0.9608]     #breathing bar color (exhale JMT_neu)
    COL_EXH_LINE = [-0.500, 0.5000, 1.0000]     #breathing cirlce line color
    COLOR_PAUSE = [0.8, 0.8, 1.0]               #lightblue
    
    
    ##  ===== CALCULATION OF PHASE TIMES =====  ##
    total_pause = PAUSE_TIME * 2
    active_time = CYCLE_TIME - total_pause
    inhale_time = active_time * INHALE_RATIO
    exhale_time = active_time - inhale_time
    
    print(f"Atemzyklus Konfiguration:")
    print(f"  Anzahl Atemzüge/Minute: {BCPM}")
    print(f"  Atemfrequenz: {1/CYCLE_TIME:.4f}Hz")
    print(f"  Dauer Atemzug: {CYCLE_TIME:.4f}s")
    print(f"  Einatmen/inhale: {inhale_time:.2f}s")
    print(f"  Ausatmen/exhale: {exhale_time:.2f}s")
    print(f" (Pausen/hold á: {PAUSE_TIME}s)")
    print(f"  Min. Größe: {MIN_SIZE_PERCENT*100:.2f}%")
    
    
    ##  create the visual components  ##
    # breathing-circle
    breathing_circle = visual.ShapeStim(
         win, name='breathing_circle',
         #size=(2*MAX_RADIUS * MIN_SIZE_PERCENT, 2*MAX_RADIUS * MIN_SIZE_PERCENT), vertices='circle',
         size=(MAX_DIAMETER * MIN_SIZE_PERCENT, MAX_DIAMETER * MIN_SIZE_PERCENT), vertices='circle',
         ori=0.0, pos=(0, 0), draggable=False, anchor='center',
         lineWidth=3.0, colorSpace='rgb', lineColor=[0, 0, 0], fillColor='white',
         opacity=None, depth=0.0, interpolate=True
     )
    
    # breathing-info/status-text
    breathing_text = visual.TextStim(
        win,
        text='',
        #pos=[0, -350],  #px - position unter dem Ball
        pos=[0, 0],  #unit: norm/px
        #height=40, #px
        height=.035, # unit: norm
        color=[1, 1, 1]
    )
    
    
    ##  ===== HELP-FUNCTIONS =====  ##
    #smoothing the circle animation
    def ease_in_out_sine(x):
        """Smoothing function for natural breathing movement/Glättungsfunktion für natürliche Atembewegung"""
        return -(np.cos(np.pi * x) - 1) / 2
    
    #check current breathing-phase, call the calculation
    def get_breathing_state(elapsed_time):
        """
        Calculates radius/diameter and color based on elapsed time
        Berechnet Radius/Durchmesser und Farbe basierend auf der verstrichenen Zeit
        
        Returns:
            #tuple: (radius, color, phase_name)
            tuple: (radius, color, line_color, phase_name)
        """
        et = elapsed_time % CYCLE_TIME
        #min_radius = MAX_RADIUS * MIN_SIZE_PERCENT
        min_diameter = MAX_DIAMETER * MIN_SIZE_PERCENT
        
        ## Phase 1: INHALE
        if et < inhale_time:
            phase = "Einatmen"
            progress = et / inhale_time
            smooth_progress = ease_in_out_sine(progress)
            #radius = min_radius + (MAX_RADIUS - min_radius) * smooth_progress
            diameter = min_diameter + (MAX_DIAMETER - min_diameter) * smooth_progress
            color = COLOR_INHALE
            line_color = COL_INH_LINE
            
        ## Phase 2: PAUSE after inhale
        elif et < inhale_time + PAUSE_TIME:
            #phase = "Halten"
            #phase = ""
            phase = "Einatmen"      #FPB, no 'hold'
            #radius = MAX_RADIUS
            diameter = MAX_DIAMETER
            #color = COLOR_PAUSE
            color = COLOR_INHALE
            line_color = COL_INH_LINE
            
        ## Phase 3: EXHALE
        elif et < inhale_time + PAUSE_TIME + exhale_time:
            phase = "Ausatmen"
            progress = (et - inhale_time - PAUSE_TIME) / exhale_time
            smooth_progress = ease_in_out_sine(progress)
            #radius = MAX_RADIUS - (MAX_RADIUS - min_radius) * smooth_progress
            diameter = MAX_DIAMETER - (MAX_DIAMETER - min_diameter) * smooth_progress
            color = COLOR_EXHALE
            line_color = COL_EXH_LINE
            
        ## Phase 4: PAUSE after exhale
        else:
            #phase = "Halten"
            #phase = ""
            phase = "Ausatmen"      #FPB, no 'hold'
            #radius = min_radius
            diameter = min_diameter
            #color = COLOR_PAUSE
            color = COLOR_EXHALE
            line_color = COL_EXH_LINE
        
        #return radius, color, phase
        return diameter, color, line_color, phase
    
    
    # --- Initialize components for Routine "thx" ---
    thx_txt = visual.TextStim(win=win, name='thx_txt',
        text='Vielen Dank,\n\ndieser Teil der Untersuchung ist zu Ende.',
        font='Open Sans',
        pos=(0, main_pos_y), draggable=False, height=0.035, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    thx_key = keyboard.Keyboard(deviceName='thx_key')
    thx_hint = visual.TextStim(win=win, name='thx_hint',
        text='Beenden mit Enter',
        font='Open Sans',
        pos=(.25, btn_pos_y), draggable=False, height=0.025, wrapWidth=None, ori=0, 
        color=[-0.25,-0.25,-0.25], colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "_settings" ---
    # create an object to store info about Routine _settings
    _settings = data.Routine(
        name='_settings',
        components=[],
    )
    _settings.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from set_things
    ##  marker: FPB (1.083Hz, 65bcpm) task START  ##
    Marker.setData(165); core.wait(0.1); Marker.setData(0)
    
    # store start times for _settings
    _settings.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    _settings.tStart = globalClock.getTime(format='float')
    _settings.status = STARTED
    thisExp.addData('_settings.started', _settings.tStart)
    _settings.maxDuration = None
    # keep track of which components have finished
    _settingsComponents = _settings.components
    for thisComponent in _settings.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_settings" ---
    _settings.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=_settings,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            _settings.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _settings.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_settings" ---
    for thisComponent in _settings.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for _settings
    _settings.tStop = globalClock.getTime(format='float')
    _settings.tStopRefresh = tThisFlipGlobal
    thisExp.addData('_settings.stopped', _settings.tStop)
    thisExp.nextEntry()
    # the Routine "_settings" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    FPB_ins_byp = data.TrialHandler2(
        name='FPB_ins_byp',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(FPB_ins_byp)  # add the loop to the experiment
    thisFPB_ins_byp = FPB_ins_byp.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFPB_ins_byp.rgb)
    if thisFPB_ins_byp != None:
        for paramName in thisFPB_ins_byp:
            globals()[paramName] = thisFPB_ins_byp[paramName]
    
    for thisFPB_ins_byp in FPB_ins_byp:
        FPB_ins_byp.status = STARTED
        if hasattr(thisFPB_ins_byp, 'status'):
            thisFPB_ins_byp.status = STARTED
        currentLoop = FPB_ins_byp
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisFPB_ins_byp.rgb)
        if thisFPB_ins_byp != None:
            for paramName in thisFPB_ins_byp:
                globals()[paramName] = thisFPB_ins_byp[paramName]
        
        # --- Prepare to start Routine "FPB_ins" ---
        # create an object to store info about Routine FPB_ins
        FPB_ins = data.Routine(
            name='FPB_ins',
            components=[FPB_ins_header, FPB_ins_txt, FPB_ins_key, FPB_ins_btn, FPB_ins_hint, FPB_ins_mse],
        )
        FPB_ins.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from FPB_ins_code
        FPB_ins_txt.alignText = 'center'
        #FPB_bcpm_ins_txt.alignText = 'center'
        
        # create starting attributes for FPB_ins_key
        FPB_ins_key.keys = []
        FPB_ins_key.rt = []
        _FPB_ins_key_allKeys = []
        # setup some python lists for storing info about the FPB_ins_mse
        FPB_ins_mse.x = []
        FPB_ins_mse.y = []
        FPB_ins_mse.leftButton = []
        FPB_ins_mse.midButton = []
        FPB_ins_mse.rightButton = []
        FPB_ins_mse.time = []
        FPB_ins_mse.clicked_name = []
        gotValidClick = False  # until a click is received
        # store start times for FPB_ins
        FPB_ins.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        FPB_ins.tStart = globalClock.getTime(format='float')
        FPB_ins.status = STARTED
        thisExp.addData('FPB_ins.started', FPB_ins.tStart)
        FPB_ins.maxDuration = None
        # keep track of which components have finished
        FPB_insComponents = FPB_ins.components
        for thisComponent in FPB_ins.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "FPB_ins" ---
        FPB_ins.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisFPB_ins_byp, 'status') and thisFPB_ins_byp.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *FPB_ins_header* updates
            
            # if FPB_ins_header is starting this frame...
            if FPB_ins_header.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FPB_ins_header.frameNStart = frameN  # exact frame index
                FPB_ins_header.tStart = t  # local t and not account for scr refresh
                FPB_ins_header.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FPB_ins_header, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FPB_ins_header.started')
                # update status
                FPB_ins_header.status = STARTED
                FPB_ins_header.setAutoDraw(True)
            
            # if FPB_ins_header is active this frame...
            if FPB_ins_header.status == STARTED:
                # update params
                pass
            
            # *FPB_ins_txt* updates
            
            # if FPB_ins_txt is starting this frame...
            if FPB_ins_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FPB_ins_txt.frameNStart = frameN  # exact frame index
                FPB_ins_txt.tStart = t  # local t and not account for scr refresh
                FPB_ins_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FPB_ins_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FPB_ins_txt.started')
                # update status
                FPB_ins_txt.status = STARTED
                FPB_ins_txt.setAutoDraw(True)
            
            # if FPB_ins_txt is active this frame...
            if FPB_ins_txt.status == STARTED:
                # update params
                pass
            
            # *FPB_ins_key* updates
            waitOnFlip = False
            
            # if FPB_ins_key is starting this frame...
            if FPB_ins_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FPB_ins_key.frameNStart = frameN  # exact frame index
                FPB_ins_key.tStart = t  # local t and not account for scr refresh
                FPB_ins_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FPB_ins_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FPB_ins_key.started')
                # update status
                FPB_ins_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(FPB_ins_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(FPB_ins_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if FPB_ins_key.status == STARTED and not waitOnFlip:
                theseKeys = FPB_ins_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
                _FPB_ins_key_allKeys.extend(theseKeys)
                if len(_FPB_ins_key_allKeys):
                    FPB_ins_key.keys = _FPB_ins_key_allKeys[-1].name  # just the last key pressed
                    FPB_ins_key.rt = _FPB_ins_key_allKeys[-1].rt
                    FPB_ins_key.duration = _FPB_ins_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *FPB_ins_btn* updates
            
            # if FPB_ins_btn is starting this frame...
            if FPB_ins_btn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FPB_ins_btn.frameNStart = frameN  # exact frame index
                FPB_ins_btn.tStart = t  # local t and not account for scr refresh
                FPB_ins_btn.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FPB_ins_btn, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FPB_ins_btn.started')
                # update status
                FPB_ins_btn.status = STARTED
                FPB_ins_btn.setAutoDraw(True)
            
            # if FPB_ins_btn is active this frame...
            if FPB_ins_btn.status == STARTED:
                # update params
                pass
            
            # *FPB_ins_hint* updates
            
            # if FPB_ins_hint is starting this frame...
            if FPB_ins_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FPB_ins_hint.frameNStart = frameN  # exact frame index
                FPB_ins_hint.tStart = t  # local t and not account for scr refresh
                FPB_ins_hint.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FPB_ins_hint, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FPB_ins_hint.started')
                # update status
                FPB_ins_hint.status = STARTED
                FPB_ins_hint.setAutoDraw(True)
            
            # if FPB_ins_hint is active this frame...
            if FPB_ins_hint.status == STARTED:
                # update params
                pass
            # *FPB_ins_mse* updates
            
            # if FPB_ins_mse is starting this frame...
            if FPB_ins_mse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FPB_ins_mse.frameNStart = frameN  # exact frame index
                FPB_ins_mse.tStart = t  # local t and not account for scr refresh
                FPB_ins_mse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FPB_ins_mse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('FPB_ins_mse.started', t)
                # update status
                FPB_ins_mse.status = STARTED
                FPB_ins_mse.mouseClock.reset()
                prevButtonState = FPB_ins_mse.getPressed()  # if button is down already this ISN'T a new click
            if FPB_ins_mse.status == STARTED:  # only update if started and not finished!
                buttons = FPB_ins_mse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(FPB_ins_btn, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(FPB_ins_mse):
                                gotValidClick = True
                                FPB_ins_mse.clicked_name.append(obj.name)
                        if not gotValidClick:
                            FPB_ins_mse.clicked_name.append(None)
                        x, y = FPB_ins_mse.getPos()
                        FPB_ins_mse.x.append(x)
                        FPB_ins_mse.y.append(y)
                        buttons = FPB_ins_mse.getPressed()
                        FPB_ins_mse.leftButton.append(buttons[0])
                        FPB_ins_mse.midButton.append(buttons[1])
                        FPB_ins_mse.rightButton.append(buttons[2])
                        FPB_ins_mse.time.append(FPB_ins_mse.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=FPB_ins,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                FPB_ins.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FPB_ins.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "FPB_ins" ---
        for thisComponent in FPB_ins.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for FPB_ins
        FPB_ins.tStop = globalClock.getTime(format='float')
        FPB_ins.tStopRefresh = tThisFlipGlobal
        thisExp.addData('FPB_ins.stopped', FPB_ins.tStop)
        # check responses
        if FPB_ins_key.keys in ['', [], None]:  # No response was made
            FPB_ins_key.keys = None
        FPB_ins_byp.addData('FPB_ins_key.keys',FPB_ins_key.keys)
        if FPB_ins_key.keys != None:  # we had a response
            FPB_ins_byp.addData('FPB_ins_key.rt', FPB_ins_key.rt)
            FPB_ins_byp.addData('FPB_ins_key.duration', FPB_ins_key.duration)
        # store data for FPB_ins_byp (TrialHandler)
        FPB_ins_byp.addData('FPB_ins_mse.x', FPB_ins_mse.x)
        FPB_ins_byp.addData('FPB_ins_mse.y', FPB_ins_mse.y)
        FPB_ins_byp.addData('FPB_ins_mse.leftButton', FPB_ins_mse.leftButton)
        FPB_ins_byp.addData('FPB_ins_mse.midButton', FPB_ins_mse.midButton)
        FPB_ins_byp.addData('FPB_ins_mse.rightButton', FPB_ins_mse.rightButton)
        FPB_ins_byp.addData('FPB_ins_mse.time', FPB_ins_mse.time)
        FPB_ins_byp.addData('FPB_ins_mse.clicked_name', FPB_ins_mse.clicked_name)
        # the Routine "FPB_ins" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisFPB_ins_byp as finished
        if hasattr(thisFPB_ins_byp, 'status'):
            thisFPB_ins_byp.status = FINISHED
        # if awaiting a pause, pause now
        if FPB_ins_byp.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            FPB_ins_byp.status = STARTED
    # completed 1.0 repeats of 'FPB_ins_byp'
    FPB_ins_byp.status = FINISHED
    
    
    # set up handler to look after randomisation of conditions etc
    FPB_byp = data.TrialHandler2(
        name='FPB_byp',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(FPB_byp)  # add the loop to the experiment
    thisFPB_byp = FPB_byp.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFPB_byp.rgb)
    if thisFPB_byp != None:
        for paramName in thisFPB_byp:
            globals()[paramName] = thisFPB_byp[paramName]
    
    for thisFPB_byp in FPB_byp:
        FPB_byp.status = STARTED
        if hasattr(thisFPB_byp, 'status'):
            thisFPB_byp.status = STARTED
        currentLoop = FPB_byp
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisFPB_byp.rgb)
        if thisFPB_byp != None:
            for paramName in thisFPB_byp:
                globals()[paramName] = thisFPB_byp[paramName]
        
        # --- Prepare to start Routine "go4FPB" ---
        # create an object to store info about Routine go4FPB
        go4FPB = data.Routine(
            name='go4FPB',
            components=[go4it_txt],
        )
        go4FPB.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for go4FPB
        go4FPB.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        go4FPB.tStart = globalClock.getTime(format='float')
        go4FPB.status = STARTED
        thisExp.addData('go4FPB.started', go4FPB.tStart)
        go4FPB.maxDuration = None
        # keep track of which components have finished
        go4FPBComponents = go4FPB.components
        for thisComponent in go4FPB.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "go4FPB" ---
        go4FPB.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisFPB_byp, 'status') and thisFPB_byp.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *go4it_txt* updates
            
            # if go4it_txt is starting this frame...
            if go4it_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                go4it_txt.frameNStart = frameN  # exact frame index
                go4it_txt.tStart = t  # local t and not account for scr refresh
                go4it_txt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(go4it_txt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'go4it_txt.started')
                # update status
                go4it_txt.status = STARTED
                go4it_txt.setAutoDraw(True)
            
            # if go4it_txt is active this frame...
            if go4it_txt.status == STARTED:
                # update params
                pass
            
            # if go4it_txt is stopping this frame...
            if go4it_txt.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > go4it_txt.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    go4it_txt.tStop = t  # not accounting for scr refresh
                    go4it_txt.tStopRefresh = tThisFlipGlobal  # on global time
                    go4it_txt.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'go4it_txt.stopped')
                    # update status
                    go4it_txt.status = FINISHED
                    go4it_txt.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=go4FPB,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                go4FPB.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in go4FPB.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "go4FPB" ---
        for thisComponent in go4FPB.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for go4FPB
        go4FPB.tStop = globalClock.getTime(format='float')
        go4FPB.tStopRefresh = tThisFlipGlobal
        thisExp.addData('go4FPB.stopped', go4FPB.tStop)
        # Run 'End Routine' code from go4FPB_code
        ##  marker: FPB (1.083Hz, 65bcpm) START  ##
        Marker.setData(65); core.wait(0.1); Marker.setData(0)
        
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if go4FPB.maxDurationReached:
            routineTimer.addTime(-go4FPB.maxDuration)
        elif go4FPB.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "FPB" ---
        # create an object to store info about Routine FPB
        FPB = data.Routine(
            name='FPB',
            components=[],
        )
        FPB.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from FPB_code
        ##  Timer für respiration (circle/cylce)  ##
        breathing_clock = core.Clock()
        
        # store start times for FPB
        FPB.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        FPB.tStart = globalClock.getTime(format='float')
        FPB.status = STARTED
        thisExp.addData('FPB.started', FPB.tStart)
        FPB.maxDuration = None
        # keep track of which components have finished
        FPBComponents = FPB.components
        for thisComponent in FPB.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "FPB" ---
        FPB.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisFPB_byp, 'status') and thisFPB_byp.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from FPB_code
            ##  updating breathing-circle parameter  ##
            while True:
                elapsed = breathing_clock.getTime()     # Berechne verstrichene Zeit
                
                ## Beende Routine nach festgelegter Zeit
                if elapsed >= BREATHING_DUR:
                    break;
                    
                ## ESC >> Exit
                if event.getKeys(['escape']):
                    break
            
                ## Hole aktuellen Zustand
                #radius, color, phase = get_breathing_state(elapsed)
                diameter, color, line_color, phase = get_breathing_state(elapsed)
            
                ## Aktualisiere Kreis | Text (optional)
                #breathing_circle.radius     = radius
                breathing_circle.size       = diameter
                breathing_circle.fillColor  = color
                breathing_circle.lineColor  = line_color
                breathing_text.text         = phase
            
                ## Zeichne die Komponenten
                breathing_circle.draw()
                breathing_text.draw()
                win.flip()
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=FPB,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                FPB.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FPB.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "FPB" ---
        for thisComponent in FPB.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for FPB
        FPB.tStop = globalClock.getTime(format='float')
        FPB.tStopRefresh = tThisFlipGlobal
        thisExp.addData('FPB.stopped', FPB.tStop)
        # Run 'End Routine' code from FPB_code
        ##  marker: FPB (1.083Hz, 65bcpm) STOP  ##
        Marker.setData(67); core.wait(0.1); Marker.setData(0)
        
        # the Routine "FPB" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisFPB_byp as finished
        if hasattr(thisFPB_byp, 'status'):
            thisFPB_byp.status = FINISHED
        # if awaiting a pause, pause now
        if FPB_byp.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            FPB_byp.status = STARTED
    # completed 1.0 repeats of 'FPB_byp'
    FPB_byp.status = FINISHED
    
    
    # --- Prepare to start Routine "thx" ---
    # create an object to store info about Routine thx
    thx = data.Routine(
        name='thx',
        components=[thx_txt, thx_key, thx_hint],
    )
    thx.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for thx_key
    thx_key.keys = []
    thx_key.rt = []
    _thx_key_allKeys = []
    # store start times for thx
    thx.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thx.tStart = globalClock.getTime(format='float')
    thx.status = STARTED
    thisExp.addData('thx.started', thx.tStart)
    thx.maxDuration = None
    # keep track of which components have finished
    thxComponents = thx.components
    for thisComponent in thx.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thx" ---
    thx.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thx_txt* updates
        
        # if thx_txt is starting this frame...
        if thx_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thx_txt.frameNStart = frameN  # exact frame index
            thx_txt.tStart = t  # local t and not account for scr refresh
            thx_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thx_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thx_txt.started')
            # update status
            thx_txt.status = STARTED
            thx_txt.setAutoDraw(True)
        
        # if thx_txt is active this frame...
        if thx_txt.status == STARTED:
            # update params
            pass
        
        # if thx_txt is stopping this frame...
        if thx_txt.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thx_txt.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                thx_txt.tStop = t  # not accounting for scr refresh
                thx_txt.tStopRefresh = tThisFlipGlobal  # on global time
                thx_txt.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thx_txt.stopped')
                # update status
                thx_txt.status = FINISHED
                thx_txt.setAutoDraw(False)
        
        # *thx_key* updates
        waitOnFlip = False
        
        # if thx_key is starting this frame...
        if thx_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thx_key.frameNStart = frameN  # exact frame index
            thx_key.tStart = t  # local t and not account for scr refresh
            thx_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thx_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thx_key.started')
            # update status
            thx_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(thx_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(thx_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if thx_key is stopping this frame...
        if thx_key.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thx_key.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                thx_key.tStop = t  # not accounting for scr refresh
                thx_key.tStopRefresh = tThisFlipGlobal  # on global time
                thx_key.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thx_key.stopped')
                # update status
                thx_key.status = FINISHED
                thx_key.status = FINISHED
        if thx_key.status == STARTED and not waitOnFlip:
            theseKeys = thx_key.getKeys(keyList=['return','space'], ignoreKeys=["escape"], waitRelease=False)
            _thx_key_allKeys.extend(theseKeys)
            if len(_thx_key_allKeys):
                thx_key.keys = _thx_key_allKeys[-1].name  # just the last key pressed
                thx_key.rt = _thx_key_allKeys[-1].rt
                thx_key.duration = _thx_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *thx_hint* updates
        
        # if thx_hint is starting this frame...
        if thx_hint.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thx_hint.frameNStart = frameN  # exact frame index
            thx_hint.tStart = t  # local t and not account for scr refresh
            thx_hint.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thx_hint, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thx_hint.started')
            # update status
            thx_hint.status = STARTED
            thx_hint.setAutoDraw(True)
        
        # if thx_hint is active this frame...
        if thx_hint.status == STARTED:
            # update params
            pass
        
        # if thx_hint is stopping this frame...
        if thx_hint.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thx_hint.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                thx_hint.tStop = t  # not accounting for scr refresh
                thx_hint.tStopRefresh = tThisFlipGlobal  # on global time
                thx_hint.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thx_hint.stopped')
                # update status
                thx_hint.status = FINISHED
                thx_hint.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=thx,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            thx.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thx.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thx" ---
    for thisComponent in thx.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thx
    thx.tStop = globalClock.getTime(format='float')
    thx.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thx.stopped', thx.tStop)
    # check responses
    if thx_key.keys in ['', [], None]:  # No response was made
        thx_key.keys = None
    thisExp.addData('thx_key.keys',thx_key.keys)
    if thx_key.keys != None:  # we had a response
        thisExp.addData('thx_key.rt', thx_key.rt)
        thisExp.addData('thx_key.duration', thx_key.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thx.maxDurationReached:
        routineTimer.addTime(-thx.maxDuration)
    elif thx.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from set_things
    ##  marker: FPB (1.083Hz, 65bcpm) task STOP  ##
    Marker.setData(167); core.wait(0.1); Marker.setData(0)
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
