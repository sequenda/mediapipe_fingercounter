import ctypes

x = 100

address = id(x)
address_hex = hex(address)

value = ctypes.c_long.from_address(address + 24).value

print(f"address: {address_hex}, value: {value}")
