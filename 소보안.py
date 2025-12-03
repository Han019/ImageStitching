#toyransom2
with open("fake_flag.enc","rb") as f_ptr:
    flag_data =f_ptr.read()

flag_key="8C9031DB7A03F189"

flag_key=bytes.fromhex(flag_key)

plain_flag=b''

for i in range(len(flag_data)):
    plain_flag += bytes([flag_key[i%len(flag_key)]^flag_data[i]])

print(plain_flag)
#toyransom3
