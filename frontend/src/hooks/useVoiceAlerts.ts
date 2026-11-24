import { useState, useEffect, useRef, useCallback } from 'react';

interface VoiceAlertOptions {
  category?: string;
  cooldownMs?: number;
  priority?: 'P1' | 'P2' | 'P3' | string;
}

export function useVoiceAlerts(defaultMuted = false) {
  const [isVoiceMuted, setIsVoiceMuted] = useState<boolean>(() => {
    try {
      const stored = localStorage.getItem('perceptguard_voice_muted');
      return stored !== null ? JSON.parse(stored) : defaultMuted;
    } catch {
      return defaultMuted;
    }
  });

  const lastSpokenRef = useRef<Map<string, number>>(new Map());

  const toggleVoiceMute = useCallback(() => {
    setIsVoiceMuted((prev) => {
      const next = !prev;
      try {
        localStorage.setItem('perceptguard_voice_muted', JSON.stringify(next));
      } catch {
        // Ignore localStorage errors
      }
      if (next && typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      return next;
    });
  }, []);

  const triggerVoiceAlert = useCallback(
    (message: string, options?: VoiceAlertOptions) => {
      if (isVoiceMuted) return;
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
      if (!message || message.trim().length === 0) return;

      const category = options?.category || message.slice(0, 24);
      const cooldownMs = options?.cooldownMs ?? 6000; // 6-second cooldown
      const now = Date.now();

      const lastTime = lastSpokenRef.current.get(category) || 0;
      if (now - lastTime < cooldownMs) {
        return; // Suppressed by cooldown
      }

      lastSpokenRef.current.set(category, now);

      try {
        // Cancel previous utterances if high priority (P1)
        if (options?.priority === 'P1') {
          window.speechSynthesis.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(message);
        utterance.rate = 1.05;
        utterance.pitch = 1.0;
        utterance.volume = 0.9;
        window.speechSynthesis.speak(utterance);
      } catch (err) {
        console.warn('[useVoiceAlerts] Speech synthesis failed:', err);
      }
    },
    [isVoiceMuted]
  );

  useEffect(() => {
    return () => {
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  return {
    isVoiceMuted,
    toggleVoiceMute,
    triggerVoiceAlert,
  };
}
