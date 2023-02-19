# Copyright by @BurnedGuitarist  |  https://www.youtube.com/@burnedguitarist/videos
# Draft script for fetching GPT response and populating SunVox project using Radiant Voices and SunVox DLL for Python.

import os
import openai
from rv.api import Project, m
from sunvox.api import Slot, deinit, init
from rv.pattern import Pattern, PatternClone
import numpy as np
import io
import re
import itertools
from sys import exit
import warnings

# Constants
SCHEMA = ["note", "velocity", "duration"]
SUNVOX_NOTES = "C c D d E F f G g A a B "
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G" "G#", "A", "A#", "B"]
OUTPUT_PROJECT_PATH = '<output sunvox project path>'

NOTE = "note"
VELOCITY = "velocity"
DURATION = "duration"

MELODY = "Melody"
DRUM = "Drum"
BASS = "Bass"

MELODY_PATTERN_NAME = "melody_pattern"
DRUM_PATTERN_NAME = "drum_pattern"
BASS_PATTERN_NAME = "bass_pattern"

PROMPT = (
    "Human: Hi GPT, could you provide me with a simple longer melody written as a SunVox pattern table? The table should have three columns: Note, Velocity, Note Duration in Ticks. Without any table boundaries. Columns should be separated by a single space, not tab. Rows should be separated by the new line character. Then in the second table write drums in the same format. Then in the third table write bass in the same format. Duration should be expressed in SunVoX pattern ticks, not seconds. Notes should be in the SunVox format, e.g. C2, G3, F1. All three tables should have the same sum of durations and be synchronized. Name the respective three table as Melody, Drum, Bass.",
)


def get_openai_response(prompt: str) -> str:
    """Get OpenAI API response for the user prompt."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"],
    )
    print(">>>>>> raw response")
    print(response)
    return response["choices"][0]["text"]


def find_substring(string: str, start: str, end: str) -> str:
    """Find substring between keywords."""
    try:
        s = string.rindex(start) + len(start)
        e = string.rindex(end, s)
        return string[s:e]
    except ValueError:
        return ""


def parse_table_to_list(data: str, permutations: list) -> list:
    """Convert text table to the array."""
    if data:
        data = [
            re.split("\t| ", row)
            for row in data.split("\n")
            if row and any(row.startswith(note) for note in permutations)
        ]
        return data


def convert_midi_to_sunvox_note(note_number: int) -> str:
    """Convert MIDI note number to the SunVox representation"""
    octave = int(note_number / 12)
    start = note_number % 12 * 2 - 2
    end = note_number % 12 * 2
    note = SUNVOX_NOTES[start:end]
    return f"{note.strip()}{octave}"


def convert_note_accidentals_to_sunvox_notation(note: str) -> int:
    """Convert SunVox note representation to MIDI integer"""
    if "#" in note:
        return f"{note[0].lower()}{note[2]}"
    else:
        return note


def convert_sunvox_note_to_midi(note: str) -> int:
    """Convert SunVox note representatin to MIDI integer"""
    note = convert_note_accidentals_to_sunvox_notation(note)
    note_letter = note[0]
    octave = int(note[1])
    idx = SUNVOX_NOTES.index(note_letter) / 2
    note_number = octave * 12 + idx + 1
    return note_number


def calculate_pattern_duration(data: list) -> int:
    """Calculate sum of instrument durations."""
    row_cnt = 0
    for row in data:
        duration = int(row[SCHEMA.index(DURATION)])
        row_cnt += duration
    return row_cnt


def generate_pattern(
    project: Project, name: str, offset: int, pattern_len: int
) -> Pattern:
    """Generate a pattern and attach it to the Project."""
    pattern = Pattern()
    pattern.lines = pattern_len
    pattern.name = name
    pattern.y = offset
    project.attach_pattern(pattern)
    return pattern


def send_events(melody_data, drum_data, bass_data):
    """Send array note data to the SunVoX pattern."""

    project = Project()

    # Melody synth path
    generator = project.new_module(m.Generator, waveform="triangle", volume=100)
    lfo = project.new_module(m.Lfo, waveform=0)
    project.connect(generator, lfo)
    distortion = project.new_module(m.Distortion, type=3, noise=60)
    project.connect(lfo, distortion)
    reverb = project.new_module(m.Reverb, wet=60)
    project.connect(distortion, reverb)
    project.connect(reverb, project.output)

    # Drum path
    reverb_drum = project.new_module(m.Reverb, wet=30)
    drumsynth = project.new_module(m.DrumSynth)
    project.connect(drumsynth, reverb_drum)
    project.connect(reverb_drum, project.output)

    # Bass path
    bass = project.new_module(
        m.Fm, mode=3, c_freq_ratio=4, m_freq_ratio=4, c_volume=102, m_volume=48
    )
    project.connect(bass, project.output)

    # Pattern alignment
    melody_pattern_len = calculate_pattern_duration(melody_data)
    drum_pattern_len = calculate_pattern_duration(drum_data)
    bass_pattern_len = calculate_pattern_duration(bass_data)
    common_pattern_len = max(melody_pattern_len, drum_pattern_len, bass_pattern_len)

    melody_pattern = generate_pattern(
        project, MELODY_PATTERN_NAME, -80, common_pattern_len
    )
    drum_pattern = generate_pattern(project, DRUM_PATTERN_NAME, 0, common_pattern_len)
    bass_pattern = generate_pattern(project, BASS_PATTERN_NAME, 80, common_pattern_len)

    project.layout()
    init(None, 44100, 2, 0)

    with Slot(project) as slot:
        slot.volume(256)

        if melody_data:
            cycles = common_pattern_len // melody_pattern_len
            pointer = 0
            for row in (x for _ in range(cycles) for x in melody_data):
                converted_note = int(
                    convert_sunvox_note_to_midi(row[SCHEMA.index(NOTE)])
                )
                velocity = int(row[SCHEMA.index(VELOCITY)])
                duration = int(row[SCHEMA.index(DURATION)])
                resp = slot.set_pattern_event(
                    slot.find_pattern(MELODY_PATTERN_NAME),
                    1,
                    pointer,
                    converted_note,
                    velocity,
                    generator.index + 1,
                    0x0000,
                    0x0000,
                ) 
                pointer += duration

        if drum_data:
            cycles = common_pattern_len // drum_pattern_len
            pointer = 0
            for row in (x for _ in range(cycles) for x in drum_data):
                if row[SCHEMA.index(NOTE)]:
                    converted_note = int(
                        convert_sunvox_note_to_midi(row[SCHEMA.index(NOTE)])
                    )
                    velocity = int(row[SCHEMA.index(VELOCITY)])
                    resp = slot.set_pattern_event(
                        slot.find_pattern(DRUM_PATTERN_NAME),
                        0,
                        pointer,
                        converted_note,
                        velocity,
                        drumsynth.index + 1,
                        0x0000,
                        0x0000,
                    ) 
                duration = int(row[SCHEMA.index(DURATION)])
                pointer += duration

        if bass_data:
            cycles = common_pattern_len // bass_pattern_len
            pointer = 0
            for row in (x for _ in range(cycles) for x in bass_data):
                if row[SCHEMA.index(NOTE)]:
                    converted_note = int(
                        convert_sunvox_note_to_midi(row[SCHEMA.index(NOTE)])
                    )
                    velocity = int(row[SCHEMA.index(VELOCITY)])
                    resp = slot.set_pattern_event(
                        slot.find_pattern(BASS_PATTERN_NAME),
                        0,
                        pointer,
                        converted_note,
                        velocity,
                        bass.index + 1,
                        0x0000,
                        0x0000,
                    ) 
                duration = int(row[SCHEMA.index(DURATION)])
                pointer += duration

        slot.stop()
        slot.save_filename(OUTPUT_PROJECT_PATH)

    deinit()


if __name__ == "__main__":
    response_text = get_openai_response(PROMPT)

    if (
        MELODY not in response_text
        or DRUM not in response_text
        or BASS not in response_text
    ):
        warnings.warn("Incomplete GPT response. Try again.")
        exit()

    octaves = range(0, 13)
    permutations = []
    for r in itertools.product(NOTES, octaves):
        permutations.append(f"{r[0]}{r[1]}")

    melody_data = parse_table_to_list(
        find_substring(response_text, MELODY, DRUM), permutations
    )
    drum_data = parse_table_to_list(
        find_substring(response_text, DRUM, BASS), permutations
    )
    bass_data = parse_table_to_list(response_text.split(BASS, 1)[1], permutations)

    if (
        not melody_data
        or not drum_data
        or not bass_data
    ):
        warnings.warn("Incomplete note table. Try again.")
        exit()

    send_events(melody_data, drum_data, bass_data)
