class Readout:
	def __init__(self, channels=0, timestamp=0, since_previous=0):
		self.channels=channels
		self.timestamp=timestamp
		self.since_previous=since_previous
		self.waveforms={}
	
	def valid(self):
		return self.channels and len(self.waveforms)==self.nChannels()
	
	def __bool__(self):
		return self.valid()
	
	def has_channel(self, c: int):
		return self.channels&(15-c)
	
	def nChannels(self):
		c = 0
		v = self.channels
		while v:
			c+=(v&1)
			v>>=1
		return c
	
	def get_waveform(self, c: int):
		if c not in waveforms:
			raise RuntimeError(f"Waveform {c} not found")
		return waveforms[c]

class EngFormatReader:
	def __init__(self, in_stream):
		self._instream=in_stream
		self.line_number=0
		self.eof=False
		self.channel_thresholds=[4096]*16
		self.coincidence_window=0
		self.buffer_all_triggers=False
		self.time_sync_words=[0]*4
	
	@staticmethod
	def clean_line(line):
		line = line.strip()
		idx = line.find('#')
		if idx!=-1:
			line=line[0:idx]
		return line
	
	def read_next_trigger(self):
		waveforms_to_read=0;
		ro=Readout()
		while True:
			self.line_number+=1
			line=self._instream.readline()
			if not line:
				self.eof=True
				return None
# 			print("Raw line:", line)
			line=EngFormatReader.clean_line(line)
# 			print("Cleaned line:", line)
			if len(line)==0:
				continue
			fields=line.split()
			if len(fields)<1:
				continue
			tag=fields[0]
			if waveforms_to_read and tag!="Waveform":
				raise ValueError(f"Unexpected record of type '{tag}' on line {self.line_number}; "
				                 f"{waveforms_to_read} more waveform(s) were expected")
			if tag=="ChannelThresholds":
				if len(fields)!=17:
					raise ValueError(f"Wrong number of fields for ChannelThresholds record on line {self.line_number}")
				try:
					self.channel_thresholds=[int(t) for t in fields[1:17]]
				except ValueError as e:
					raise ValueError(e.args[0]+f" for channel threshold value on line {self.line_number}")
			elif tag=="CoincidenceWindow":
				if len(fields)!=2:
					raise ValueError(f"Wrong number of fields for CoincidenceWindow record on line {self.line_number}")
				try:
					self.coincidence_window=int(fields[1])
				except ValueError as e:
					raise ValueError(e.args[0]+f" for coincidence window value on line {self.line_number}")
			elif tag=="BufferAllTriggers":
				if len(fields)!=2:
					raise ValueError(f"Wrong number of fields for BufferAllTriggers record on line {self.line_number}")
				if fields[1].lower()=="true" or fields[1]=="1":
					self.buffer_all_triggers=True
				elif fields[1].lower()=="false" or fields[1]=="0":
					self.buffer_all_triggers=False
				else:
					raise ValueError(f"Invalid boolean value '{fields[1]}' for BufferAllTriggers record on line {self.line_number}")
			elif tag=="TimeSync":
				if len(fields)!=5:
					raise ValueError(f"Wrong number of fields for TimeSync record on line {self.line_number}")
				try:
					self.time_sync_words=[int(w,base=16) for w in fields[1:5]]
				except ValueError as e:
					raise ValueError(e.args[0]+f" for time sync word on line {self.line_number}")
			elif tag=="Trigger":
				if len(fields)!=4:
					raise ValueError(f"Wrong number of fields for Trigger record on line {self.line_number}")
				try:
					ro.channels=int(fields[1],base=16)
				except:
					raise ValueError(f"Malformed Trigger record on line {self.line_number}: Invalid channel mask")
				waveforms_to_read=ro.nChannels()
# 				print("waveforms_to_read is now",waveforms_to_read)
				try:
					ro.timestamp=int(fields[2])
				except:
					raise ValueError(f"Malformed Trigger record on line {self.line_number}: Invalid timestamp")
				try:
					ro.since_previous=int(fields[3])
				except:
					raise ValueError(f"Malformed Trigger record on line {self.line_number}: Invalid number of samples since previous trigger")
			elif tag=="Waveform":
				if len(fields)<3:
					raise ValueError(f"Too few fields for Waveform record on line {self.line_number}")
				try:
					channel=int(fields[1])
				except:
					raise ValueError(f"Malformed Waveform record on line {self.line_number}: Invalid channel number")
				try:
					n_samples=int(fields[2])
				except:
					raise ValueError(f"Malformed Waveform record on line {self.line_number}: Invalid number of waveform samples")
				if len(fields)!=3+n_samples:
					raise ValueError(f"Wrong number of fields for Waveform record on line {self.line_number}")
				try:
					ro.waveforms[channel] = [int(i) for i in fields[3:]]
				except:
					raise ValueError(f"Malformed Waveform record on line {self.line_number}: invalid or missing waveform sample {i}")
					i+=1
				waveforms_to_read-=1
# 				print("waveforms_to_read is now",waveforms_to_read)
				if waveforms_to_read==0:
					break
			else:
				raise ValueError(f"Unrecognized record tag '{tag}' on line {self.line_number}")
		return ro