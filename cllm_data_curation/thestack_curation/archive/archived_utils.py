# import time
# import requests
# from concurrent.futures import ThreadPoolExecutor
# import asyncio
# import aiohttp
# import nest_asyncio
#
# nest_asyncio.apply()
#
# # Semaphore to limit the number of simultaneous connections
# semaphore = asyncio.Semaphore(8)  # Adjust this value based on your system's limits
#
#
# async def download_file(url, dest_folder, bearer_token):
#     headers = {
#         'Authorization': f'Bearer {bearer_token}'
#     }
#     async with semaphore:
#         async with aiohttp.ClientSession(headers=headers) as session:
#             async with session.get(url) as response:
#                 filename = os.path.join(dest_folder, url.split('/')[-2], url.split('/')[-1])
#                 time.sleep(0.05)
#                 with open(filename, 'wb') as f:
#                     while True:
#                         chunk = await response.content.read(1024)
#                         if not chunk:
#                             break
#                         f.write(chunk)
#
#
# async def async_parallel_download(urls, dest_folder, bearer_token):
#     tasks = [download_file(url, dest_folder, bearer_token) for url in urls]
#     await asyncio.gather(*tasks)
#
#
# # # Define the download function def download_file(url, dest_folder, bearer_token): headers = {'Authorization':
# # f'Bearer {bearer_token}'} response = requests.get(url, headers=headers) filename = os.path.join(dest_folder,
# # url.split('/')[-2], url.split('/')[-1]) if not os.path.isdir(os.path.join(dest_folder, url.split('/')[-2])):
# # os.makedirs(os.path.join(dest_folder, url.split('/')[-2]), exist_ok=True) with open(filename, 'wb') as f: f.write(
# # response.content)
#
# # # Define the parallel download function
# # def parallel_download(urls, dest_folder, bearer_token, max_workers=5):
# #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
# #         futures = [executor.submit(download_file, url, dest_folder, bearer_token) for url in urls]
# #         for future in futures:
# #             future.result()
#
# # Replace 'urls' with a list of download URLs, 'dest_folder' with the destination folder,
# # 'bearer_token' with the Bearer token, and 'max_workers' with the desired level of parallelism
# # urls = x["train"]
# dest_folder = '/home/paperspace/nb_in/inputs'
# bearer_token = "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"
# max_workers = 8
#
# # Run the asynchronous download
# asyncio.run(async_parallel_download(urls, dest_folder, bearer_token))