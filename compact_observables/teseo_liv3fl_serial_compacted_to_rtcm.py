import serial
import time
import numpy as np
from rtcm import MsmType, encode_msm
from rtcm.msm.domain import Observable, ObservableId, Signal

from ntrip_server import send
from ntrip_server.domain import Caster, Credentials

from bitarray import bitarray
from bitarray.util import int2ba, zeros, ba2int

# Constants
GPS_C = 299792458  # speed of light
RANGE_MS = GPS_C * 0.001  # range in 1 ms

# Teseo LIV3FL constants
FRONT_END_FREQ = {
    '0': -47122.395833492279,
    '1': -40526.315789699554,
}

PSTMTS_IDX = {
    'type': 0, 'dsp-dat': 1, 'sat-ID': 2, 'PsR': 3, 'Freq': 4,
    'cp': 5, 'flags': 6, 'CN0': 7, 'ttim': 8,
    'satdat': 9, 'Satx': 10, 'Saty': 11, 'Satz': 12,
    'Velx': 13, 'Vely': 14, 'Velz': 15,
    'src': 16, 'ac': 17, 'rrc': 18, 'pr_delta': 19,
    'cp_delta': 20, 'difdat': 21, 'drc': 22, 'drrc>': 23,
    'predav1': 24, 'predage': 25, 'predeph': 26,
}

PSTMTG_IDX = {
    'type': 0, 'week': 1, 'TOW': 2, 'TotSat': 3, 'CPUTime': 4, 'Timevalid': 5, 'NCO': 6, 'config_status': 7,
    'tow_delta': 8, 'req_tow_delta': 9, 'cpu_time_p': 10, 'req_cpu_time': 11,
    'constellation_mask': 12, 'time_best_sat_type': 13, 'time_master_sat_type': 14, 'time_aux_sat_type': 15,
    'time_master_week_n': 16, 'time_master_tow': 17, 'time_master_validity': 18, 'time_aux_week_n': 19,
    'time_aux_tow': 20, 'time_aux_validity': 21
}

# Encoding constants
# OPTION A.4
TIME_BITS = 10
SAT_ID_BITS = 5
N_MS_BITS = 5
CODE_PHASE_BITS = 18
MAX_SAT = 6
REF_N_MS = 60

# Define serial connection
ser = serial.Serial(
    port='COM5', \
    baudrate=9600, \
    parity=serial.PARITY_NONE, \
    stopbits=serial.STOPBITS_ONE, \
    bytesize=serial.EIGHTBITS, \
    timeout=1)

# -------------------------------------------------------------------
## READ DATA FROM RECEIVER

# Restart receiver in COLD mode
ser.write("$PSTMCOLD\r".encode())
print('Receiver started in cold mode')
# Timeout to make sure the restart is carried out before reading data
timeout = time.time() + 3
while True:
    ser.readline()
    if time.time() > timeout:
        break

n_lines = 200  # maximum number of lines to be read
obs_message_list = list()  # list to store observables $PSTMTS messages
corr_sat = 0  # number of correct satellite observables read
print('Reading data from the receiver')
for idx in range(0, n_lines):
    time_message = ser.readline().decode().split(',')
    print(time_message)
    if time_message[PSTMTG_IDX['type']] == '$PSTMTG':
        if time_message[PSTMTG_IDX['Timevalid']] in ['2', '7', '9', '10']:
            while True:
                obs_message = ser.readline().decode().split(',')
                print(obs_message)
                if obs_message[PSTMTS_IDX['type']] == '$PSTMTS':
                    psr = float(obs_message[PSTMTS_IDX['PsR']])
                    pr_delta = float(obs_message[PSTMTS_IDX['pr_delta']])
                    actual_pr = psr + pr_delta
                    n_ms_pr = actual_pr / RANGE_MS // 1 - REF_N_MS
                    flags = int(obs_message[PSTMTS_IDX['flags']])
                    if n_ms_pr >= 0 and n_ms_pr < 32 and flags % 2:
                        obs_message_list.append(obs_message)
                        corr_sat += 1
                elif obs_message[PSTMTG_IDX['type']] == '$PSTMTG':
                    if corr_sat >= 6:
                        print(corr_sat,' correct satellite data found, stop reading')
                        break
                    else:
                        print('Only ',corr_sat, ' satellite data found, continue reading')
                        time_message = obs_message
                        obs_message_list = list()
                        corr_sat = 0
                        continue
            break

ser.close()

# -------------------------------------------------------------------
## COMPUTE TOW AND PSEUDORANGE

# Compute time of week (TOW)
tow = float(time_message[PSTMTG_IDX['TOW']])
tow_delta = float(time_message[PSTMTG_IDX['tow_delta']])
actual_tow = tow + tow_delta / GPS_C
config_status = time_message[PSTMTG_IDX['config_status']]

#Observable array
# Each row corresponds to a satellite entry
# Each column contains: 0: prn, 1: pseudorange, 2; cn0, 3: 1 ms integer part, 4: 1 ms range
obs_array = np.zeros((len(obs_message_list), 5))

# Store data in observable array
for idx, message in enumerate(obs_message_list):
    prn = int(message[PSTMTS_IDX['sat-ID']])
    psr = float(message[PSTMTS_IDX['PsR']])
    freq = float(message[PSTMTS_IDX['Freq']])
    cp = float(message[PSTMTS_IDX['cp']])
    cn0 = float(message[PSTMTS_IDX['CN0']])
    pr_delta = float(message[PSTMTS_IDX['pr_delta']])
    cp_delta = float(message[PSTMTS_IDX['cp_delta']])
    actual_pr = psr + pr_delta
    actual_cp = -(cp + cp_delta)
    doppler = freq - FRONT_END_FREQ[str(int(config_status))[1]]

    obs_array[idx, 0] = prn
    obs_array[idx, 1] = actual_pr / RANGE_MS
    obs_array[idx, 2] = cn0


# -------------------------------------------------------------------
## COMPACT GNSS DATA TO 20 BYTES PAYLOAD
# Encode observables to 20 bytes
# Order observable array by CN0 and filter first MAX_SAT satellites
obs_array = obs_array[obs_array[:, 2].argsort()[::-1]][:MAX_SAT]
# Compute 1 ms range
obs_array[:, 4] = obs_array[:, 1] % 1
# Find the element with lowest 1 ms range value and move it to the top
min_1ms_row_idx = np.argmin(obs_array[:, 4])
obs_array[[0, min_1ms_row_idx]] = obs_array[[min_1ms_row_idx, 0]]
# Remove first 1 ms range from all 1 ms range to obtain a 0 ms range
obs_array[:, 1] = obs_array[:, 1] - obs_array[0, 4]
# Compute and encode 1 ms integer part
obs_array[:, 3] = obs_array[:, 1] // 1 - REF_N_MS
# Compute 1 ms range
obs_array[:, 4] = obs_array[:, 1] % 1
# Encode 1 ms range according to CODE_PHASE_BITS
obs_array[:, 4] = np.round(obs_array[:, 4] / 2 ** -CODE_PHASE_BITS)

# Encode time
tow_seconds = round(actual_tow)
time_encoded = int2ba(tow_seconds)[-TIME_BITS:]

message_20bytes = bitarray()
message_20bytes += time_encoded
# Loop over satellites ordered by CN0
for idx in range(0, MAX_SAT):
    prn_encoded = int2ba(int(obs_array[idx, 0] - 1), SAT_ID_BITS)
    message_20bytes += prn_encoded
    if idx == 0:
        message_20bytes += int2ba(int(obs_array[idx, 3]), N_MS_BITS)
    else:
        message_20bytes += int2ba(int(obs_array[idx, 3]), N_MS_BITS)
        message_20bytes += int2ba(int(obs_array[idx, 4]), CODE_PHASE_BITS)


print('Payload size: ', len(message_20bytes),'bits = ',len(message_20bytes)/8,'bytes')

# Decode from 20 bytes to observables

# Decode time
time_decoded = ba2int(message_20bytes[:TIME_BITS])
system_time = tow_seconds + 10
system_time_decoded = ba2int(int2ba(system_time)[-TIME_BITS:])
time_difference = np.mod(system_time_decoded - time_decoded, 2 ** TIME_BITS)
tow_decoded = system_time - time_difference
decoded_tow_milliseconds = round(tow_decoded * 1e3)

# Decode pseudorange
gps_decoded_list = list()

FIRST_SAT_LEN = TIME_BITS + SAT_ID_BITS + N_MS_BITS
OTHER_SAT_LEN = SAT_ID_BITS + N_MS_BITS + CODE_PHASE_BITS
for idx in range(0, MAX_SAT):
    if idx == 0:
        prn_decoded = ba2int(message_20bytes[TIME_BITS:TIME_BITS + SAT_ID_BITS]) + 1
        pr_decoded = ba2int(message_20bytes[TIME_BITS + SAT_ID_BITS:TIME_BITS + SAT_ID_BITS + N_MS_BITS]) + REF_N_MS
    else:
        prn_decoded = ba2int(
            message_20bytes[
            FIRST_SAT_LEN + (idx - 1) * OTHER_SAT_LEN:FIRST_SAT_LEN + (idx - 1) * OTHER_SAT_LEN + SAT_ID_BITS]) + 1
        pr_rough_decoded = ba2int(message_20bytes[
                                  FIRST_SAT_LEN + (idx - 1) * OTHER_SAT_LEN + SAT_ID_BITS:FIRST_SAT_LEN + (
                                          idx - 1) * OTHER_SAT_LEN + SAT_ID_BITS + N_MS_BITS]) + REF_N_MS
        pr_fine_decoded = ba2int(message_20bytes[
                                 FIRST_SAT_LEN + (idx - 1) * OTHER_SAT_LEN + SAT_ID_BITS + N_MS_BITS:FIRST_SAT_LEN + (
                                         idx - 1) * OTHER_SAT_LEN + SAT_ID_BITS + N_MS_BITS + CODE_PHASE_BITS])
        pr_fine_decoded = pr_fine_decoded * 2 ** -CODE_PHASE_BITS
        pr_decoded = pr_rough_decoded + pr_fine_decoded

    # Store in Observable domain
    gps_decoded_list.append(Observable(
        observable_id=ObservableId(prn_decoded, Signal.GPS_L1CA),
        pseudo_range=pr_decoded * RANGE_MS,
        carrier_phase=pr_decoded * RANGE_MS,
        cn0=40))

# Send to AlbaSpot Lite caster
# Encode MSM5 message
decoded_rtcm_bin = encode_msm(gps_decoded_list, MsmType.MSM5, 0, decoded_tow_milliseconds)
# Define NTRIP Caster
caster_albaspot_lite = Caster("caster.demo.albaspot.albora.xyz", 2101, Credentials("adria01", "AlbaspotCaster2021"))
# Sent RTCM data to caster
send(decoded_rtcm_bin, "adria01", caster_albaspot_lite)


