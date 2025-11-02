export const INPUT_DEVICE_TYPES = {
  KEYROW: "keyrow",
  NUMPAD: "numpad",
};

const KEYROW_PHRASES = [
  "Pack my box with five dozen liquor jugs.",
  "How vexingly quick daft zebras jump!",
  "Sphinx of black quartz, judge my vow.",
  "Waltz, bad nymph, for quick jigs vex.",
  "Crazy Frederick bought many very exquisite opal jewels.",
  "Just keep typing every quirky phrase you know by heart.",
];

const NUMPAD_PHRASES = [
  "159 357 951 753 159",
  "2580 147 369 8520",
  "7410 852 963 0741",
  "963 852 741 0 258",
  "4862 7305 9184 6270",
  "102938 4756 1029",
];

export const PHRASES = {
  [INPUT_DEVICE_TYPES.KEYROW]: KEYROW_PHRASES,
  [INPUT_DEVICE_TYPES.NUMPAD]: NUMPAD_PHRASES,
};

export const getDeviceTypeLabel = (type) => {
  switch (type) {
    case INPUT_DEVICE_TYPES.NUMPAD:
      return "Numeric keypad";
    case INPUT_DEVICE_TYPES.KEYROW:
    default:
      return "Top-row number keys";
  }
};

export const getRandomPhrase = (deviceType, currentPhrase = "") => {
  const pool = PHRASES[deviceType] || PHRASES[INPUT_DEVICE_TYPES.KEYROW];
  if (!pool || pool.length === 0) {
    return currentPhrase;
  }

  const candidates = pool.filter((phrase) => phrase !== currentPhrase);
  const options = candidates.length > 0 ? candidates : pool;
  const index = Math.floor(Math.random() * options.length);
  return options[index];
};
