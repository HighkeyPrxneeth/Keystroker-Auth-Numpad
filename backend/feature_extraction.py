import logging
from typing import List, Optional, Tuple

import numpy as np
from models import InputDeviceType, KeystrokeEvent, KeystrokePattern

logger = logging.getLogger(__name__)


class KeystrokeFeatureExtractor:
    """Extract behavioral features from keystroke timing data"""

    def __init__(self):
        self.feature_names = []

    def extract_features(
        self,
        events: List[KeystrokeEvent],
        text_typed: str,
        input_device_type: Optional[InputDeviceType] = None,
    ) -> KeystrokePattern:
        """
        Extract comprehensive features from keystroke events

        Features extracted:
        1. Hold Times (dwell time) - time between keydown and keyup for each key
        2. Down-Down Times (flight time) - time between consecutive keydown events
        3. Up-Down Times - time between keyup of one key and keydown of next
        4. Statistical measures - mean, std, skewness, rhythm patterns
        """

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x.timestamp)

        # Separate keydown and keyup events
        keydown_events = [e for e in sorted_events if e.event_type == "keydown"]
        keyup_events = [e for e in sorted_events if e.event_type == "keyup"]

        # Extract timing features
        hold_times = self._calculate_hold_times(keydown_events, keyup_events)
        down_down_times = self._calculate_down_down_times(keydown_events)
        up_down_times = self._calculate_up_down_times(keyup_events, keydown_events)

        # Calculate typing metrics
        total_time = self._calculate_total_typing_time(sorted_events)
        typing_speed = len(text_typed) / total_time if total_time > 0 else 0
        rhythm_variance = np.var(down_down_times) if down_down_times else 0

        # Create feature vector
        feature_vector, feature_names = self._create_feature_vector(
            hold_times, down_down_times, up_down_times, typing_speed, rhythm_variance
        )

        device_features, device_feature_names = self._calculate_device_usage_metrics(
            keydown_events
        )
        feature_vector.extend(device_features)
        feature_names.extend(device_feature_names)

        return KeystrokePattern(
            user_id="",  # Will be set by caller
            text_typed=text_typed,
            raw_events=events,
            input_device_type=input_device_type,
            hold_times=hold_times,
            down_down_times=down_down_times,
            up_down_times=up_down_times,
            total_typing_time=total_time,
            typing_speed=typing_speed,
            rhythm_variance=rhythm_variance,
            feature_vector=feature_vector,
            feature_names=feature_names,
        )

    def _calculate_hold_times(
        self, keydown_events: List[KeystrokeEvent], keyup_events: List[KeystrokeEvent]
    ) -> List[float]:
        """Calculate dwell time for each key press"""
        hold_times = []

        # Create mapping of key_code to keydown timestamps to handle repeated keys
        keydown_map = {}
        for event in keydown_events:
            keydown_map.setdefault(event.key_code, []).append(event.timestamp)

        for keyup_event in keyup_events:
            timestamps = keydown_map.get(keyup_event.key_code)
            if timestamps:
                down_timestamp = timestamps.pop(0)
                hold_time = keyup_event.timestamp - down_timestamp
                if hold_time > 0:  # Filter out negative times (timing errors)
                    hold_times.append(hold_time)

        return hold_times

    def _calculate_down_down_times(
        self, keydown_events: List[KeystrokeEvent]
    ) -> List[float]:
        """Calculate flight time between consecutive key presses"""
        down_down_times = []

        for i in range(1, len(keydown_events)):
            flight_time = keydown_events[i].timestamp - keydown_events[i - 1].timestamp
            if flight_time > 0:
                down_down_times.append(flight_time)

        return down_down_times

    def _calculate_up_down_times(
        self, keyup_events: List[KeystrokeEvent], keydown_events: List[KeystrokeEvent]
    ) -> List[float]:
        """Calculate time between key release and next key press"""
        up_down_times = []

        # Sort all events by timestamp
        all_events = sorted(keyup_events + keydown_events, key=lambda x: x.timestamp)

        for i in range(len(all_events) - 1):
            current_event = all_events[i]
            next_event = all_events[i + 1]

            if (
                current_event.event_type == "keyup"
                and next_event.event_type == "keydown"
            ):
                up_down_time = next_event.timestamp - current_event.timestamp
                if up_down_time > 0:
                    up_down_times.append(up_down_time)

        return up_down_times

    def _calculate_total_typing_time(self, events: List[KeystrokeEvent]) -> float:
        """Calculate total time from first to last keystroke"""
        if len(events) < 2:
            return 0
        return events[-1].timestamp - events[0].timestamp

    def _create_feature_vector(
        self,
        hold_times: List[float],
        down_down_times: List[float],
        up_down_times: List[float],
        typing_speed: float,
        rhythm_variance: float,
    ) -> Tuple[List[float], List[str]]:
        """Create comprehensive feature vector for ML models"""
        features = []
        feature_names = []

        # Statistical measures for hold times
        if hold_times:
            features.extend(
                [
                    np.mean(hold_times),
                    np.std(hold_times),
                    np.median(hold_times),
                    np.percentile(hold_times, 25),
                    np.percentile(hold_times, 75),
                    self._calculate_skewness(hold_times),
                ]
            )
            feature_names.extend(
                [
                    "hold_time_mean",
                    "hold_time_std",
                    "hold_time_median",
                    "hold_time_q1",
                    "hold_time_q3",
                    "hold_time_skewness",
                ]
            )
        else:
            features.extend([0] * 6)
            feature_names.extend(
                [
                    "hold_time_mean",
                    "hold_time_std",
                    "hold_time_median",
                    "hold_time_q1",
                    "hold_time_q3",
                    "hold_time_skewness",
                ]
            )

        # Statistical measures for down-down times
        if down_down_times:
            features.extend(
                [
                    np.mean(down_down_times),
                    np.std(down_down_times),
                    np.median(down_down_times),
                    np.percentile(down_down_times, 25),
                    np.percentile(down_down_times, 75),
                    self._calculate_skewness(down_down_times),
                ]
            )
            feature_names.extend(
                [
                    "dd_time_mean",
                    "dd_time_std",
                    "dd_time_median",
                    "dd_time_q1",
                    "dd_time_q3",
                    "dd_time_skewness",
                ]
            )
        else:
            features.extend([0] * 6)
            feature_names.extend(
                [
                    "dd_time_mean",
                    "dd_time_std",
                    "dd_time_median",
                    "dd_time_q1",
                    "dd_time_q3",
                    "dd_time_skewness",
                ]
            )

        # Statistical measures for up-down times
        if up_down_times:
            features.extend(
                [
                    np.mean(up_down_times),
                    np.std(up_down_times),
                    np.median(up_down_times),
                    np.percentile(up_down_times, 25),
                    np.percentile(up_down_times, 75),
                    self._calculate_skewness(up_down_times),
                ]
            )
            feature_names.extend(
                [
                    "ud_time_mean",
                    "ud_time_std",
                    "ud_time_median",
                    "ud_time_q1",
                    "ud_time_q3",
                    "ud_time_skewness",
                ]
            )
        else:
            features.extend([0] * 6)
            feature_names.extend(
                [
                    "ud_time_mean",
                    "ud_time_std",
                    "ud_time_median",
                    "ud_time_q1",
                    "ud_time_q3",
                    "ud_time_skewness",
                ]
            )

        # Overall typing characteristics
        features.extend([typing_speed, rhythm_variance])
        feature_names.extend(["typing_speed", "rhythm_variance"])

        # Rhythm patterns (autocorrelation features)
        if down_down_times and len(down_down_times) > 1:
            autocorr = self._calculate_autocorrelation(down_down_times)
            features.append(autocorr)
            feature_names.append("rhythm_autocorr")
        else:
            features.append(0)
            feature_names.append("rhythm_autocorr")

        return features, feature_names

    def _calculate_device_usage_metrics(
        self, keydown_events: List[KeystrokeEvent]
    ) -> Tuple[List[float], List[str]]:
        """Generate device-specific features such as key ratios and switch counts."""

        numpad_keycodes = set(range(96, 106))  # Numpad digits 0-9
        keyrow_keycodes = set(range(48, 58))  # Top-row digits 0-9

        total_keydowns = len(keydown_events)
        numpad_count = 0
        keyrow_count = 0
        device_switches = 0

        last_device: Optional[str] = None
        digit_keycodes: List[int] = []

        for event in keydown_events:
            device: Optional[str] = None
            if event.key_code in numpad_keycodes:
                numpad_count += 1
                device = InputDeviceType.NUMPAD.value
                digit_keycodes.append(event.key_code)
            elif event.key_code in keyrow_keycodes:
                keyrow_count += 1
                device = InputDeviceType.KEYROW.value
                digit_keycodes.append(event.key_code)

            if device:
                if last_device and last_device != device:
                    device_switches += 1
                last_device = device

        digit_total = numpad_count + keyrow_count
        non_digit_keydowns = total_keydowns - digit_total

        numpad_ratio_all = numpad_count / total_keydowns if total_keydowns else 0.0
        keyrow_ratio_all = keyrow_count / total_keydowns if total_keydowns else 0.0
        numpad_ratio_digits = numpad_count / digit_total if digit_total else 0.0
        keyrow_ratio_digits = keyrow_count / digit_total if digit_total else 0.0
        switch_rate = device_switches / max(1, digit_total - 1)

        mean_digit_keycode = float(np.mean(digit_keycodes)) if digit_keycodes else 0.0

        features = [
            numpad_ratio_all,
            keyrow_ratio_all,
            numpad_ratio_digits,
            keyrow_ratio_digits,
            float(device_switches),
            switch_rate,
            float(numpad_count),
            float(keyrow_count),
            float(non_digit_keydowns),
            mean_digit_keycode,
        ]

        feature_names = [
            "numpad_ratio_all_keydowns",
            "keyrow_ratio_all_keydowns",
            "numpad_ratio_digit_keydowns",
            "keyrow_ratio_digit_keydowns",
            "device_switch_count",
            "device_switch_rate",
            "numpad_keydown_count",
            "keyrow_keydown_count",
            "non_digit_keydown_count",
            "digit_keycode_mean",
        ]

        return features, feature_names

    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of the data"""
        if len(data) < 3:
            return 0

        mean_val = np.mean(data)
        std_val = np.std(data)

        if std_val == 0:
            return 0

        skew = np.mean([((x - mean_val) / std_val) ** 3 for x in data])
        return skew

    def _calculate_autocorrelation(self, data: List[float], lag: int = 1) -> float:
        """Calculate autocorrelation for rhythm analysis"""
        if len(data) <= lag:
            return 0

        data_array = np.array(data)
        n = len(data_array)

        # Normalize data
        data_normalized = data_array - np.mean(data_array)

        # Calculate autocorrelation
        autocorr = np.correlate(data_normalized, data_normalized, mode="full")
        autocorr = autocorr[n - 1 :]

        if len(autocorr) > lag and autocorr[0] != 0:
            return autocorr[lag] / autocorr[0]

        return 0


def validate_keystroke_data(events: List[KeystrokeEvent], min_events: int = 6) -> bool:
    """Validate that keystroke data is sufficient for feature extraction"""
    if len(events) < min_events:
        return False

    # Check for both keydown and keyup events
    keydown_count = sum(1 for e in events if e.event_type == "keydown")
    keyup_count = sum(1 for e in events if e.event_type == "keyup")

    if keydown_count == 0 or keyup_count == 0:
        return False

    # Check for reasonable timing (events should span at least 500ms)
    if len(events) > 1:
        time_span = max(e.timestamp for e in events) - min(e.timestamp for e in events)
        if time_span < 500:  # Less than 500ms
            return False

    return True
