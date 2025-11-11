export const INPUT_DEVICE_TYPES = {
  KEYROW: "keyrow",
  NUMPAD: "numpad",
};

// Combined pool of all literary phrases
const ALL_PHRASES = [
  {
    text: "He was born on the 4th of July, 1984, at 12:01 AM, weighing 7 pounds and 3 ounces.",
    source: "The Last Thing He Told Me by Laura Dave",
  },
  {
    text: "The access code is 8675309, repeated twice for a total of 14 digits of security.",
    source: "Ready Player One by Ernest Cline",
  },
  {
    text: "My phone number is 212-555-0187, and my P.O. Box is 81901.",
    source: "Extremely Loud & Incredibly Close by Jonathan Safran Foer",
  },
  {
    text: "Her flight confirmation was 3B489X, departing at 0600 from gate C9.",
    source: "The Da Vinci Code by Dan Brown",
  },
  {
    text: "Remember, the combination to the safe is 42-13-87-09-21-5.",
    source: "The Girl with the Dragon Tattoo by Stieg Larsson",
  },
  {
    text: "The GPS coordinates are 40.7128 N, 74.0060 W, marking the center of the city.",
    source: "Digital Fortress by Dan Brown",
  },
  {
    text: "The project number is 778923-B, with a budget of $15,345,987.52.",
    source: "The Andromeda Strain by Michael Crichton",
  },
  {
    text: "The target frequency is precisely 1,420,405,751.768 hertz, the hydrogen line.",
    source: "Contact by Carl Sagan",
  },
  {
    text: "His new serial number was 76435-993-7364-01-B, a long and meaningless string.",
    source: "Fahrenheit 451 by Ray Bradbury",
  },
  {
    text: "The case file, #88790-A, detailed the events of 03/17/2003, including witness statements 1 through 12.",
    source: "The Lincoln Lawyer by Michael Connelly",
  },
  {
    text: "The bank account, 007-4-876543-2, held the entirety of the stolen 5,000,000 francs.",
    source: "The Da Vinci Code by Dan Brown",
  },
  {
    text: "Ledger entry 9-15-1797: 4 barrels, 3 crates, value $1,275.50, sold at dock No. 2.",
    source: "Moby-Dick by Herman Melville",
  },
  {
    text: "At 11:59 P.M. on 12-31-1899, the clock struck 12 and the new century began.",
    source: "The Time Machine by H. G. Wells",
  },
  {
    text: "Entry dated Dec 25, 1890: Case #221B-12; jewel valued $1000; suspect at 11:47.",
    source: "The Adventure of the Blue Carbuncle by Arthur Conan Doyle",
  },
  {
    text: "Jean Valjean, prisoner 24601, served 19 yrs from 1796 to 1815 in Toulon.",
    source: "Les Misérables by Victor Hugo",
  },
  {
    text: "At 8:32 P.M. on 07-12-1843, Scrooge counted 2 pennies and 3 candles left burning.",
    source: "A Christmas Carol by Charles Dickens",
  },
  {
    text: "Holmes's telegram, 12-10-1894, 10:45 A.M.: 'Meet me at 221B, case #14-B-7.'",
    source: "The Memoirs of Sherlock Holmes by Arthur Conan Doyle",
  },
];

// Get a random phrase from the combined pool
export const getRandomPhrase = (currentPhrase = "") => {
  if (ALL_PHRASES.length === 0) {
    return { text: currentPhrase, source: "" };
  }

  // Extract current text if it's an object
  const currentText =
    typeof currentPhrase === "string"
      ? currentPhrase
      : currentPhrase.text || currentPhrase;

  const candidates = ALL_PHRASES.filter(
    (phrase) => phrase.text !== currentText
  );
  const options = candidates.length > 0 ? candidates : ALL_PHRASES;
  const index = Math.floor(Math.random() * options.length);
  return options[index];
};

// Alias for authentication (same as getRandomPhrase now)
export const getRandomAuthenticationPhrase = getRandomPhrase;

// Utility functions for device type labels
export const getDeviceTypeLabel = (type) => {
  switch (type) {
    case INPUT_DEVICE_TYPES.NUMPAD:
      return "Numeric keypad";
    case INPUT_DEVICE_TYPES.KEYROW:
    default:
      return "Top-row number keys";
  }
};

export const getDeviceGuidance = (type) => {
  if (type === INPUT_DEVICE_TYPES.NUMPAD) {
    return "Use the dedicated numeric keypad on the right-hand side for every digit; keep Num Lock on.";
  }
  return "Use the top-row digits (0-9) above the letter keys; avoid the numpad entirely.";
};
