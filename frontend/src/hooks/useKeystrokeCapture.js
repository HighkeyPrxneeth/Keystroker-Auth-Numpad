import { useState, useCallback, useRef } from 'react';

export const useKeystrokeCapture = () => {
  const [keystrokeData, setKeystrokeData] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const startTimeRef = useRef(null);
  const keyDownTimesRef = useRef({});

  const startCapture = useCallback(() => {
    setKeystrokeData([]);
    setIsCapturing(true);
    startTimeRef.current = Date.now();
    keyDownTimesRef.current = {};
  }, []);

  const stopCapture = useCallback(() => {
    setIsCapturing(false);
    startTimeRef.current = null;
    keyDownTimesRef.current = {};
  }, []);

  const handleKeyDown = useCallback((event) => {
    if (!isCapturing) return;
    
    const key = event.key;
    const timestamp = Date.now();
    
    // Store the key down time
    keyDownTimesRef.current[key] = timestamp;
  }, [isCapturing]);

  const handleKeyUp = useCallback((event) => {
    if (!isCapturing || !startTimeRef.current) return;
    
    const key = event.key;
    const timestamp = Date.now();
    const keyDownTime = keyDownTimesRef.current[key];
    
    if (keyDownTime) {
      const dwellTime = timestamp - keyDownTime;
      const relativeTime = timestamp - startTimeRef.current;
      
      setKeystrokeData(prev => [...prev, {
        key,
        keyCode: event.keyCode,
        timestamp: relativeTime,
        dwellTime,
      }]);
      
      // Clean up the key down time
      delete keyDownTimesRef.current[key];
    }
  }, [isCapturing]);

  const getFlightTimes = useCallback(() => {
    const flightTimes = [];
    for (let i = 1; i < keystrokeData.length; i++) {
      const flightTime = keystrokeData[i].timestamp - keystrokeData[i-1].timestamp;
      flightTimes.push(flightTime);
    }
    return flightTimes;
  }, [keystrokeData]);

  const getTypingPattern = useCallback(() => {
    return {
      keystrokes: keystrokeData,
      flightTimes: getFlightTimes(),
      totalTime: keystrokeData.length > 0 ? 
        keystrokeData[keystrokeData.length - 1].timestamp : 0,
      typingSpeed: keystrokeData.length > 0 ? 
        (keystrokeData.length / (keystrokeData[keystrokeData.length - 1].timestamp / 1000)) * 60 : 0
    };
  }, [keystrokeData, getFlightTimes]);

  return {
    keystrokeData,
    isCapturing,
    startCapture,
    stopCapture,
    handleKeyDown,
    handleKeyUp,
    getTypingPattern,
  };
};
