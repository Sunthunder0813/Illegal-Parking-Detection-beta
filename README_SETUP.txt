How to Use Raspberry Pi as a Media Relay for IP Cameras (Accessible from Railway or Internet)
============================================================================================

1. Connect your Raspberry Pi and IP cameras to the same router/network.

2. Start the Flask app on your Raspberry Pi:
   $ python3 app.py

3. Access the web interface locally:
   Open http://<raspi-ip>:5000/ in a browser on any device on the same network.

4. To make the Pi accessible from Railway or the public internet, you have two options:

   Option A: Port Forwarding (Router Configuration)
   ------------------------------------------------
   - Log in to your router's admin page.
   - Forward port 5000 to your Pi's local IP (e.g., 192.168.18.32:5000).
   - Find your public IP (e.g., https://whatismyipaddress.com/).
   - Now your Pi is accessible at http://<your-public-ip>:5000/
   - WARNING: Exposing your Pi to the internet has security risks. Use strong passwords and firewall.

   Option B: Use ngrok (Recommended for testing)
   ---------------------------------------------
   - Install ngrok on your Pi:
     $ curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
     $ echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
     $ sudo apt update && sudo apt install ngrok
   - Start a tunnel:
     $ ngrok http 5000
   - Copy the public URL ngrok gives you (e.g., https://xxxx.ngrok.io).
   - Your Pi is now accessible at that URL.

5. Update your Railway frontend/server to use the Pi's public address:
   - Set the backend/server IP in your Railway app to the Pi's public IP or ngrok URL.
   - All API and video requests from Railway should go through the Pi, not directly to the cameras.

6. Test the connection:
   - From Railway or any remote device, try to access:
     http://<public-pi-address>:5000/video_feed_c1
     http://<public-pi-address>:5000/api/server_status
   - You should see the video stream and get a JSON response if the server is online.

7. Security Note:
   - Exposing your Pi to the internet can be risky. For production, use authentication, firewall, or VPN.

8. Summary of the architecture:
   [IP Cameras] <--LAN--> [Raspberry Pi Flask App] <--Internet--> [Railway Frontend]

9. If you want to update the Pi's IP automatically for Railway, use a public service (Pastebin, JSONBin, etc.) or a tunnel like ngrok.

10. For any changes, always restart your Flask app on the Pi.

