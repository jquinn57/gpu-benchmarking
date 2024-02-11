import asyncio
from kasa import SmartStrip, SmartPlug
# pip install python-kasa
# run on command line to discover devices on local network: kasa discover


async def update():
    kasa_ip = '192.168.1.87'
    plug_name = 'workstation'
    ss = SmartStrip(kasa_ip)
    await ss.update()
    sp = ss.get_plug_by_name(plug_name)
    for n in range(20):
        power = await sp.get_emeter_realtime()
        print(power.power)
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(update())
