import time
from brping import Ping1D,Ping360,definitions

p = Ping360()
p.connect_serial('/dev/ttyUSB0',115200)
# p.connect_udp(host, int(port))
print("Initialized: %s" % p.initialize())
print(p.set_transmit_frequency(800))
print(p.set_sample_period(80))
print(p.set_number_of_samples(200))
tstart_s = time.time()
for x in range(400):
    p.transmitAngle(x)
tend_s = time.time()
print(p)
print("full scan in %dms, %dHz" % (1000*(tend_s - tstart_s), 400/(tend_s - tstart_s)))
# turn on auto-scan with 1 grad steps
p.control_auto_transmit(0,399,1,20)
tstart_s = time.time()
# wait for 400 device_data messages to arrive
for x in range(400):
    p.wait_message([definitions.PING360_DEVICE_DATA])
tend_s = time.time()

print("full scan in %dms, %dHz" % (1000*(tend_s - tstart_s), 400/(tend_s - tstart_s)))

# stop the auto-transmit process
p.control_motor_off()

# turn on auto-transmit with 10 grad steps
p.control_auto_transmit(0,399,10,20)

tstart_s = time.time()
# wait for 40 device_data messages to arrive (40 * 10grad steps = 400 grads)
for x in range(40):
    p.wait_message([definitions.PING360_DEVICE_DATA])
tend_s = time.time()

print("full scan in %dms, %dHz" % (1000*(tend_s - tstart_s), 400/(tend_s - tstart_s)))

p.control_reset(0, 0)