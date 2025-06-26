class SoundDS_val(Dataset):
  def __init__(self, df, data_path):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 50000
    self.sr = 44100
    self.channel = 2
    self.shift_pct = 0.4

  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.df)

  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):
    try:
      audio_file = self.data_path + self.df.loc[idx, 'relative_path']
      # Get the Class ID
      class_id = self.df.loc[idx, 'classID']

      aud = AudioUtil.open(audio_file)

      reaud = AudioUtil.resample(aud, self.sr)
      rechan = AudioUtil.rechannel(reaud, self.channel)

      dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
  #     shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
      sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
      aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

      return aug_sgram, class_id
    except Exception as e:
        print(f"[ERROR] Fallo en idx={idx} | Archivo: {self.df.loc[idx, 'relative_path']} | Error: {e}")
        return None