import os
import discord
from dotenv import load_dotenv
from discord.ext.commands import Bot
from discord.ext import commands
import saberfaceImageClassifier
import numpy as np
import requests

load_dotenv()
TOKEN = os.getenv('WASHINGTON_BOT')

bot = commands.Bot(command_prefix='s!')

#Setup and load the trained model to be used with the discord bot
model = saberfaceImageClassifier.create_model()
model_path = os.getenv('CHECKPOINT_PATH_V2') 
model.load_weights(model_path)
print("Model loaded")


#Establish the on ready method that handles the event when discord has established a connection with the bot
@bot.event
async def on_ready():   
    print(f'{bot.user} is online!')

#Command used to see if an attached image is an image of a saberface or not
@bot.command(name="saberface")
async def saberface(ctx):
    #Get the attachment size to check that an attachment has indeed been uploaded
    attachment_size = len(ctx.message.attachments)
    if attachment_size == 1:
        image_url = ctx.message.attachments[0].url
        #Download the attachment with requests
        img_data = requests.get(image_url).content
        with open('input_image.jpg', 'wb') as handler:
            handler.write(img_data)
            
        #Load the attachment to predict on with the bot
        image_path = "V:\Code\SaberfaceML\input_image.jpg"
        img = saberfaceImageClassifier.keras.preprocessing.image.load_img(
            image_path, target_size=(saberfaceImageClassifier.img_height, saberfaceImageClassifier.img_width)
        )
        img_array = saberfaceImageClassifier.keras.preprocessing.image.img_to_array(img) 
        img_array = saberfaceImageClassifier.tf.expand_dims(img_array, 0) # Create a batch

        #Make the prediction
        predictions = model.predict(img_array)
        score = saberfaceImageClassifier.tf.nn.softmax(predictions[0])
    
        output_string = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(saberfaceImageClassifier.class_names[np.argmax(score)], 100 * np.max(score))
        
        if "Not_Saberfaces" in output_string:
            await ctx.send(output_string.replace("most likely belongs to Not_Saberfaces","most likely contains no saberfaces"))
        else:
            await ctx.send(output_string.replace("most likely belongs to Saberfaces","most likely contains a saberface"))
    
    else:
        await ctx.send("Provide an image!")

bot.run(TOKEN)