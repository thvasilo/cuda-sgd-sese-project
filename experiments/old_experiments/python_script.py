from subprocess import Popen, PIPE

Nbatch = '10'
batch_size = '10'
Ndata = '1000'
cmd = './gradient'

p = Popen([cmd, batch_size, Ndata], stdout=PIPE)
output = p.stdout.read()
lines = output.splitlines()
time = lines[0].split('=')
blocks = lines[1].split('=')
threads = lines[2].split('=')

