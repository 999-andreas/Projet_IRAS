"""
Interface de base pour la gestion des événements de scan.
Permet d'abstraire l'affichage/la notification (CLI, WebSocket, logs...).
"""


class ScanEvents:
    """
    Interface abstraite pour la propagation des événements du scanner SDR.
    Toutes les méthodes ont des implémentations vides par défaut
    pour éviter de forcer la surcharge des événements non utilisés.
    """

    def on_scan_start(self, info: dict):
        """Appelé au démarrage global du scan avec la configuration."""
        pass

    def on_hop_complete(self, hop_index: int, total_hops: int, freq_hz: float,
                         floor_db: float, best_snr_db: float, signals: list,
                         saturation: bool, hop_plot_path: str):
        """Appelé lorsqu'un saut de fréquence (hop) est terminé et analysé."""
        pass

    def on_signal_extracted(self, freq_hz: float, snr_db: float, bw_hz: float,
                            wav_path: str, debug_plot_path: str):
        """Appelé lorsqu'un signal a été filtré, extrait et sauvegardé."""
        pass

    def on_signal_duplicate(self, freq_hz: float):
        """Appelé lorsqu'un signal détecté est rejeté car déjà connu."""
        pass

    def on_hop_error(self, hop_index: int, total_hops: int, freq_hz: float, error: Exception):
        """Appelé si une erreur survient pendant l'acquisition/analyse d'un hop."""
        pass

    def on_scan_end(self, detected_count: int, total_hops: int, output_dir: str):
        """Appelé à la fin complète du balayage."""
        pass

    def on_scan_interrupted(self):
        """Appelé si le scan est interrompu par l'utilisateur (stop_flag ou Ctrl+C)."""
        pass


class CliScanEvents(ScanEvents):
    """
    Implémentation console (CLI) optionnelle.
    """

    def on_scan_start(self, info: dict):
        print(f"\n[SCAN STARTED] Range: {info.get('freq_start', 0)/1e6:.2f} - "
              f"{info.get('freq_end', 0)/1e6:.2f} MHz")

    def on_hop_complete(self, hop_index, total_hops, freq_hz, floor_db, best_snr_db, signals, saturation, hop_plot_path):
        sat_str = " [SATURATION]" if saturation else ""
        print(f"[{hop_index}/{total_hops}] {freq_hz/1e6:.3f} MHz | "
              f"Floor: {floor_db:.1f} dBFS | SNR: {best_snr_db:.1f} dB | "
              f"Signals: {len(signals)}{sat_str}")

    def on_signal_extracted(self, freq_hz, snr_db, bw_hz, wav_path, debug_plot_path):
        print(f"  └── [SIGNAL] {freq_hz/1e6:.4f} MHz (SNR: {snr_db:.1f} dB, BW: {bw_hz/1e3:.1f} kHz) -> {wav_path}")

    def on_signal_duplicate(self, freq_hz):
        print(f"  └── [DUPLICATE] {freq_hz/1e6:.4f} MHz ignoré")

    def on_hop_error(self, hop_index, total_hops, freq_hz, error):
        print(f"[{hop_index}/{total_hops}] {freq_hz/1e6:.3f} MHz | ERREUR: {error}")

    def on_scan_end(self, detected_count, total_hops, output_dir):
        print(f"\n[SCAN FINISHED] {detected_count} signaux sauvegardés dans {output_dir}")

    def on_scan_interrupted(self):
        print("\n[SCAN INTERRUPTED]")