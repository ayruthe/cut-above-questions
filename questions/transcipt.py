from datasets import load_dataset, Dataset
import json
import os
from pathlib import Path
from typing import List
import sys


class TransciptDataset:
    def __init__(self, filepath: Path):
        """Dataset class to handle transcript information from interview data.

        TranscriptDataset stores the metadata as lists of strings and stores the questions,
        answers, and named entities as lists of lists.

        Attributes:
            file: transcipt file location
            title: podcast episode title
            subtitle: 
            description:
            host:
            guest:
            named_entities:
            questions:
            answers:
        """
        self.file: List[str] = []
        self.title: List[str] = []
        self.subtitle: List[str] = []
        self.description: List[str] = []
        self.host: List[str] = []
        self.guest: List[str] = []
        # self.named_entities: List[list] = []
        self.questions: List[list] = []
        self.answers: List[list] = []
        
        files = os.listdir(filepath)
        for file in files:
            self.parse_file(filepath.joinpath(file))
            self.file.append(file)

        self.to_json()
    

    def parse_file(self, file: str):
        """Description.
        """
        transcript_data = {}

        with open(file) as f:
            lines = f.readlines()
            guest_labeled = False
            curr_speaker = None
            conversation_started = False
            curr_dialogue = ""
            host_questions: List[str] = []
            guest_answers: List[str] = []
            for line in lines:

                if not guest_labeled:
                    if 'TITLE:' in line and 'SUBTITLE:' not in line:
                        self.title.append(line.split('TITLE: ')[-1].split('\n')[0])
                    elif 'SUBTITLE:' in line:
                        self.subtitle.append(line.split('SUBTITLE: ')[-1].split('\n')[0])
                    elif 'DESCRIPTION:' in line:
                        self.description.append(line.split('DESCRIPTION: ')[-1].split('\n')[0])
                    elif 'Host:' in line:
                        host_firstname, host_lastname = line.split(' ')[1:3]
                        host_fullname = host_firstname + ' ' + host_lastname
                        self.host.append(host_fullname)
                    elif 'Guest:' in line:
                        guest_firstname, guest_lastname = line.split(' ')[1:3]
                        guest_fullname = guest_firstname + ' ' + guest_lastname
                        self.guest.append(guest_fullname)
                        curr_speaker = guest_fullname
                        guest_labeled = True

                if 'music\n' in line:
                    continue

                if line in ['\t', '\n', ' \n', 'music\n']:
                    continue

                if ('[' in line) and (']' in line):
                    continue

                if guest_labeled:
                    if ((guest_fullname + ':' in line) or (guest_firstname + ':' in line) or ('OUTRO' in line)) and ('Guest:' not in line):
                        if curr_speaker == host_fullname:
                            if len(curr_dialogue) > 0:
                                host_questions.append(curr_dialogue)
                            del curr_dialogue
                            curr_dialogue = ""
                        curr_dialogue += line.split(guest_fullname + ': ')[-1]
                        curr_speaker = guest_fullname

                    elif (host_fullname + ':' in line) or (host_firstname + ':' in line) or ('INTRO' in line) or ('OUTRO' in line):
                        # If host found and guest is current speaker, save and reset buffer. 
                        # Otherwise, the host may be talking twice and this should be extended.
                        if curr_speaker == guest_fullname:
                            if len(curr_dialogue) > 0:
                                guest_answers.append(curr_dialogue)
                            del curr_dialogue
                            curr_dialogue = ""
                        curr_dialogue += line.split(host_fullname + ': ')[-1]
                        curr_speaker = host_fullname
                        conversation_started = True
                
                    elif conversation_started:
                        curr_dialogue += line

        if len(host_questions) > len(guest_answers):
            if len(host_questions) - len(guest_answers) == 1:
                host_questions.pop(len(host_questions)-1)
            else:
                sys.exit(f'Host Questions Lenght: {len(host_questions)}, Guest Answers Length: {len(guest_answers)}!')
        self.questions.append(host_questions)
        self.answers.append(guest_answers)


    def to_json(self):

        # r = {'is_claimed': 'True', 'rating': 3.5}
        # r = json.dumps(r)
        # loaded_r = json.loads(r)
        # loaded_r['rating'] #Output 3.5
        # type(r) #Output str
        # type(loaded_r) #Output dict
        pass