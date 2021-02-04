"""
Agent that gets the local keyboard input in the act() function.

Applies safety classifier(s) to process user and partner messages.
"""

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.utils.misc import display_messages
from parlai.utils.strings import colorize
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier


OFFENSIVE_USER_REPLY = '[ Sorry, could not process that message. Please try again. ]'
OFFENSIVE_BOT_REPLY = (
    '[ Unsafe model reply detected. Generating a new model reply. ]'
)

NO_MATCH = "no_match"
STRING_MATCHED = "string_matched"
CLASSIFIER_MATCHED = "classifier_matched"


class SafetyAgent:
    def __init__(self, opt, safety_classifier=None):
        self.id = 'safeLocalHuman'
        self._init_safety(opt, safety_classifier)

    def _init_safety(self, opt, safety_classifier=None):
        """
        Initialize safety modules.
        """
        if opt['safety'] == 'string_matcher' or opt['safety'] == 'all':
            self.offensive_string_matcher = OffensiveStringMatcher()
        if opt['safety'] == 'classifier' or opt['safety'] == 'all':
            if safety_classifier:
                self.offensive_classifier = safety_classifier
            else:
                self.offensive_classifier = OffensiveLanguageClassifier()

    def check_offensive(self, text):
        """
        Check if text is offensive using string matcher and classifier.
        """
        if text == '':
            return NO_MATCH
        if (
            hasattr(self, 'offensive_string_matcher')
            and text in self.offensive_string_matcher
        ):
            return STRING_MATCHED
        if hasattr(self, 'offensive_classifier') and text in self.offensive_classifier:
            return CLASSIFIER_MATCHED

        return NO_MATCH

    def observe_and_act(self, msg):
        """
        Observe bot reply if and only if it passes.
        """
        # Now check if bot was offensive
        is_offensive = self.check_offensive(msg)
        return is_offensive
