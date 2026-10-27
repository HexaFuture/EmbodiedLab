"""Measured mechanical controls drive continuous electrical states and a scope."""
import hashlib
import math
import numpy as np
from .solver import Circuit
from .submission import digest

SELECTORS=('CAP','LOAD','MODE')
DETENT_LABELS={'CAP':('C100','C220','C470'),'LOAD':('150','704','3300'),'MODE':('ON','OFF','DISCHARGE')}
DETENT_SPACING_M=.030
CHANNELS=['time_s','source_voltage_V','output_voltage_V','load_current_A','source_current_A']

def load_from_position(position):
    if not math.isfinite(position):raise ValueError('invalid load encoder')
    return 150.*22.**(np.clip(position,0.,.06)/.06)

def detent(name,p):
    k=round(p/.03)
    return DETENT_LABELS[name][k] if 0<=k<=2 and abs(p-k*.03)<=.004 else None

class RectifierBench:
    def __init__(self,episode_id,configuration):
        self.model=Circuit(configuration['capacitances_F'],configuration['diode_n'])
        self.episode_id=episode_id;self.tick=0;self.active=None;self.records={};self.actions=[];self.state_log=[]
        self.positions={n:DETENT_LABELS[n].index(configuration['initial'][n])*.03 for n in SELECTORS}
        self.switch=(configuration['initial']['CAP'],load_from_position(self.positions['LOAD']),configuration['initial']['MODE'])
        self.latest=np.zeros(5);self.last_record=0

    def step(self,tick,positions):
        if tick!=self.tick+1:raise RuntimeError('electrical clock discontinuity')
        cap,r,mode=self.switch
        rows=self.model.advance(100,None if cap is None else DETENT_LABELS['CAP'].index(cap),r,mode)
        self.last_rows=rows
        self.latest=rows[-1].copy();self.tick=tick
        if self.active is not None:
            # 12 kHz sampled/quantized scope; 240 Hz control changes at tick boundaries.
            samples=rows[1::2].copy()
            samples[:,1:3]=np.round(samples[:,1:3]/.002)*.002
            samples[:,3:5]=np.round(samples[:,3:5]/.0001)*.0001
            record=self.records[self.active]
            record['chunks'].append(samples)
            record['controls'].append(dict(start_tick=tick-1,cap=cap,mode=mode,
                load_encoder_m=round(round(self.positions['LOAD']/.00001)*.00001,5),
                load_ohm=round(float(load_from_position(round(self.positions['LOAD']/.00001)*.00001)),3)))
            if np.any(np.abs(samples[:,1:3])>20) or np.any(np.abs(samples[:,3:5])>1):record['overrange']=True
            if len(record['chunks'])>=240*30:self.stop()   # a record ends by itself after 30 s
        self.positions=dict(positions)
        switch=(detent('CAP',positions['CAP']),float(load_from_position(positions['LOAD'])),detent('MODE',positions['MODE']))
        if switch[0]!=self.switch[0] or switch[2]!=self.switch[2] or abs(switch[1]-self.switch[1])>.01:
            self.actions.append(dict(tick=tick,old=list(self.switch),new=list(switch)))
            self.state_log.append(dict(tick=tick,capacitor_voltages_V=self.model.voltages.tolist(),switch=list(switch)))
        self.switch=switch

    def reading(self):
        p=round(self.positions['LOAD']/.00001)*.00001
        return dict(physics_tick=self.tick,cap=self.switch[0],mode=self.switch[2],load_encoder_m=round(p,5),
                    load_ohm=round(float(load_from_position(p)),3),output_voltage_V=round(float(self.latest[2])/.002)*.002,
                    active_evidence_id=self.active)

    def start(self):
        if self.active is not None:raise ValueError('scope already recording')
        self.last_record+=1;key=f'{self.episode_id}-scope-{self.last_record}'
        self.records[key]=dict(chunks=[],controls=[],start_tick=self.tick,end_tick=None,overrange=False,start_controls=self.reading())
        self.active=key;return key

    def stop(self):
        if self.active is None:raise ValueError('scope is not recording')
        key=self.active;record=self.records[key];record['end_tick']=self.tick;record['end_controls']=self.reading()
        record['samples']=np.concatenate(record.pop('chunks')).tolist() if self.tick>record['start_tick'] else []
        record['sha256']=digest(self.record_payload(key))
        self.active=None;return key

    def record_payload(self,key):
        r=self.records[key]
        return dict(evidence_id=key,channels=CHANNELS,sample_hz=12000,start_tick=r['start_tick'],end_tick=r['end_tick'],
                    start_controls=r['start_controls'],end_controls=r.get('end_controls'),overrange=r['overrange'],
                    controls=r['controls'],samples=r.get('samples',[]))

    def process_extra(self):
        """The scope record still recording at stop (samples and controls so far; None when idle), for the
        offline process score. Never shown to the Agent."""
        if self.active is None:return {'scope_open':None}
        r=self.records[self.active]
        return {'scope_open':dict(evidence_id=self.active,status='open',channels=CHANNELS,sample_hz=12000,
            start_tick=r['start_tick'],last_tick=self.tick,start_controls=dict(r['start_controls']),overrange=r['overrange'],
            controls=[dict(c) for c in r['controls']],
            samples=np.concatenate(r['chunks']).tolist() if r['chunks'] else [])}

    def export(self,key,offset=0,limit=512):
        if key not in self.records or self.records[key]['end_tick'] is None:raise ValueError('complete evidence required')
        if type(offset) is not int or type(limit) is not int or offset<0 or not 1<=limit<=512:raise ValueError('invalid scope page')
        r=self.records[key]
        return dict(evidence_id=key,channels=CHANNELS,sample_hz=12000,start_tick=r['start_tick'],end_tick=r['end_tick'],
                    start_controls=r['start_controls'],end_controls=r['end_controls'],overrange=r['overrange'],
                    total_samples=len(r['samples']),offset=offset,samples=r['samples'][offset:offset+limit],
                    sha256=r['sha256'],control_samples=r['controls'][offset//50:(offset+limit+49)//50],
                    hash_format='SHA256 canonical JSON of full record: sorted keys, separators comma/colon, finite numbers')
