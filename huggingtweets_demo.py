#@title ⠀ {display-mode: "form"}

def print_html(x):
    "Better printing"
    x = x.replace('\n', '<br>')
    display(HTML(x))
        
# Check we use GPU
import torch
from IPython.display import display, HTML, Javascript, clear_output
if not torch.cuda.is_available():
    print_html('Error: GPU was not found\n1/ click on the "Runtime" menu and "Change runtime type"\n'\
          '2/ set "Hardware accelerator" to "GPU" and click "save"\n3/ click on the "Runtime" menu, then "Run all" (below error should disappear)')
    raise ValueError('No GPU available')
else:
    # Install dependencies
    !pip install wandb transformers torch -qq

    import ipywidgets as widgets
    from IPython import get_ipython
    import json
    import urllib3
    import random
    import wandb
    wandb.login(anonymous='allow')  # ensure we log with huggingface
    from transformers import (
        AutoConfig, AutoTokenizer, AutoModelWithLMHead,
        TextDataset, DataCollatorForLanguageModeling,
        Trainer, TrainingArguments)
    
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    log_debug = widgets.Output()

    def fix_text(text):
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        return text

    def html_table(data, title=None):
        'Create a html table'
        width_twitter = '75px'
        def html_cell(i, twitter_button=False):
            return f'<td style="width:{width_twitter}">{i}</td>' if twitter_button else f'<td>{i}</td>'
        def html_row(row):
            return f'<tr>{"".join(html_cell(r, not i if len(row)>1 else False) for i,r in enumerate(row))}</tr>'    
        body = f'<table style="width:100%">{"".join(html_row(r) for r in data)}</table>'
        title_html = f'<h3>{title}</h3>' if title else ''
        html = '''
        <html>
            <head>
                <style>
                    table {border-collapse: collapse !important;}
                    td {text-align:left !important; border: solid #E3F2FD !important; border-width: 1px 0 !important; padding: 6px !important;}
                    tr:nth-child(even) {background-color: #E3F2FD !important;}
                </style>
            </head>
            <body>''' + title_html + body + '</body></html>'
        return(html)
        
    def cleanup_tweet(tweet):
        "Clean tweet text"
        text = ' '.join(t for t in tweet.split() if 'http' not in t)
        if text.split() and text.split()[0] == '.':
            text = ' '.join(text.split()[1:])
        return text

    def boring_tweet(tweet):
        "Check if this is a boring tweet"
        boring_stuff = ['http', '@', '#', 'thank', 'thanks', 'I', 'you']
        if len(tweet.split()) < 3:
            return True
        if all(any(bs in t.lower() for bs in boring_stuff) for t in tweet):
            return True
        return False

    def dl_tweets():
        handle_widget.disabled = True
        run_dl_tweets.disabled = True
        run_dl_tweets.button_style = 'primary'
        handle = handle_widget.value.strip()
        handle = handle[1:] if handle[0] == '@' else handle
        handle = handle.lower()
        log_dl_tweets.clear_output(wait=True)

        try_success = False

        with log_dl_tweets:
            try:
                print_html(f'\nDownloading {handle_widget.value.strip()} tweets... This should take no more than a minute!')
                http = urllib3.PoolManager(retries=urllib3.Retry(3))
                res = http.request("GET", f"http://us-central1-playground-111.cloudfunctions.net/tweets_http?handle={handle}")
                curated_tweets = json.loads(res.data.decode('utf-8'))
                curated_tweets = [fix_text(tweet) for tweet in curated_tweets]
                log_dl_tweets.clear_output(wait=True)
                print_html(f'\n{len(curated_tweets)} tweets from {handle_widget.value.strip()} downloaded!\n\n')
                    
                # create dataset
                clean_tweets = [cleanup_tweet(t) for t in curated_tweets]
                cool_tweets = [tweet for tweet in clean_tweets if not boring_tweet(tweet)]
                total_text = '\n'.join(cool_tweets)
                
                # display a few tweets
                random.shuffle(curated_tweets)
                example_tweets = [[t] for  t in curated_tweets[-8:]]
                display(HTML(html_table(example_tweets)))

                if len(total_text) < 5000:
                    # need about 4000 chars for one data sample (but depends on spaces, etc)
                    raise ValueError('Error: this user does not have enough tweets to train a Neural Network')

                if len(total_text) < 30000:
                    print_html('\n<b>Warning: this user does not have many tweets which may impact the results of the Neural Network</b>')

                with open(f'data_{handle}_train.txt', 'w') as f:
                    f.write(total_text)
                
                run_dl_tweets.button_style = 'success'
                log_finetune.clear_output(wait=True)
                run_finetune.disabled = False

                try_success = True

            except Exception as e:
                print('\nAn error occured...\n')
                print(e)
                run_dl_tweets.button_style = 'danger'
        
        if try_success:
            log_finetune.clear_output(wait=True)
            with log_finetune:
                print_html('\nFine-tune your model by clicking on "Train Neural Network"')
                
        handle_widget.disabled = False
        run_dl_tweets.disabled = False
                
    handle_widget = widgets.Text(value='@elonmusk',
                                placeholder='Enter twitter handle')

    run_dl_tweets = widgets.Button(
        description='Download tweets',
        button_style='primary')
    def on_run_dl_tweets_clicked(b):
        dl_tweets()
    run_dl_tweets.on_click(on_run_dl_tweets_clicked)

    log_restart = widgets.Output()
    log_dl_tweets = widgets.Output()
        
    # Associate run to a project
    with log_debug:
        %env WANDB_PROJECT=huggingtweets
        %env WANDB_WATCH=false
        %env WANDB_ENTITY=wandb
        %env WANDB_ANONYMOUS=allow
        %env WANDB_NOTEBOOK_NAME=huggingtweets-demo
        %env WANDB_RESUME=allow
        %env WANDB_NOTES=Github repo: https://github.com/borisdayma/huggingtweets

    # Have global access to model & tokenizer
    trainer, tokenizer = None, None
    
    def finetune():
        if run_finetune.button_style == 'success':
            # user double clicked before start of function
            return

        handle_widget.disabled = True
        run_dl_tweets.disabled = True
        run_finetune.disabled = True
        run_finetune.button_style = 'primary'
        handle = handle_widget.value.strip()
        handle = handle[1:] if handle[0] == '@' else handle
        handle = handle.lower()
        log_finetune.clear_output(wait=True)
        clear_output(wait=True)

        success_try = False

        with log_finetune:
            print_html(f'\nTraining Neural Network on {handle_widget.value.strip()} tweets... This could take up to 2-3 minutes!\n')
            progress = widgets.FloatProgress(value=0.1, min=0.0, max=1.0, bar_style = 'info')
            display(progress)

        with log_debug:
            try:
                # use new run id
                run_id = wandb.util.generate_id()
                %env WANDB_RUN_ID=$run_id
                run_name = handle_widget.value.strip()
                %env WANDB_NAME=$run_name
                wandb.init(config={'version':0.1})
                
                # Setting up pre-trained neural network
                with log_finetune:
                    print_html('\nSetting up pre-trained neural network...')
                global trainer, tokenizer
                config = AutoConfig.from_pretrained('gpt2')
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
                model = AutoModelWithLMHead.from_pretrained('gpt2', config=config)
                block_size = tokenizer.max_len
                train_dataset = TextDataset(tokenizer=tokenizer, file_path=f'data_{handle}_train.txt', block_size=block_size, overwrite_cache=True)
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                epochs = 4  # limit before overfitting
                training_args = TrainingArguments(
                    output_dir=f'output/{handle}',
                    overwrite_output_dir=True,
                    do_train=True,
                    num_train_epochs=epochs,
                    per_gpu_train_batch_size=1,
                    logging_steps=5,
                    save_steps=0,
                    seed=random.randint(0,2**32-1))
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    prediction_loss_only=True)
                progress.value = 0.4
                
                p_start, p_end = 0.4, 1.
                def progressify(f):
                    "Control progress bar when calling f"
                    def inner(*args, **kwargs):
                        if trainer.epoch is not None:
                            progress.value = p_start + trainer.epoch / epochs * (p_end - p_start)
                        return f(*args, **kwargs)
                    return inner
        
                trainer._training_step = progressify(trainer._training_step)
                
                # Training neural network
                with log_finetune:
                    print_html('Training neural network...\n')
                    display(wandb.jupyter.Run())
                    print_html('\n')
                    display(progress)
                trainer.train()

                run_finetune.button_style = 'success'
                run_predictions.disabled = False

                progress.value = 1.0
                progress.bar_style = 'success'
                success_try = True

                with log_finetune:
                    print_html('\n🎉 Neural network trained successfully!')
                log_predictions.clear_output(wait=True)
                with log_predictions:
                    print_html('\nEnter the start of a sentence and click "Run predictions"')
                with log_restart:
                    print_html('\n<b>To change user, click on menu "Runtime" → "Restart and run all"</b>\n')

            except Exception as e:
                print('\nAn error occured...\n')
                print(e)
                run_finetune.button_style = 'danger'
                run_finetune.disabled = False
                            
        if not success_try:
            display(log_debug)
            progress.bar_style = 'danger'
        
    run_finetune = widgets.Button(
        description='Train Neural Network',
        button_style='primary',
        disabled=True)
    def on_run_finetune_clicked(b):
        finetune()
    run_finetune.on_click(on_run_finetune_clicked)

    log_finetune = widgets.Output()
    with log_finetune:
        print_html('\nWaiting for Step 1 to complete...')

    def clean_prediction(text):
        token = '<|endoftext|>'
        while len(token)>1:
            text = text.replace(token, '')
            token = token[:-1]
        text = text.strip()
        if text[-1] == '"' and text.count('"') % 2: text = text[:-1]
        return text.strip()

    predictions = []
    
    def predict():
        run_predictions.disabled = True
        start_widget.disabled = True
        run_predictions.button_style = 'primary'
        handle = handle_widget.value.strip()
        handle = handle[1:] if handle[0] == '@' else handle
        handle_uncased = handle
        handle = handle.lower()
        log_predictions.clear_output(wait=True)

        # tweet buttons don't appear well in colab if within log_predictions widget
        # we reset the entire cell
        clear_output(wait=True)
        display(widgets.VBox([start_widget, run_predictions, log_predictions]))

        def tweet_html(text):
            max_char = 239
            text = text.replace('"', '&quot;')
            tweet_text = f'Trained a neural network on @{handle_uncased}: {start} → {text}'
            
            if len(tweet_text) > max_char:
                # shorten tweet
                n_words = len(tweet_text.split())
                while len(tweet_text) > max_char:
                    tweet_text = ' '.join(tweet_text.split()[:-1]) + '…'

            return '<a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-size="large" '\
                    f'data-text="{tweet_text}" '\
                    f'data-url="{wandb_url}" data-hashtags="huggingtweets" data-related="borisdayma,weights_biases,huggingface"'\
                    'data-show-count="false">Tweet</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>'
        
        success_try = False

        # get start sentence
        get_ipython().kernel.do_one_iteration() # widget slow to update on phones
        start = start_widget.value.strip()
                
        with log_predictions:
            print_html(f'\nPerforming predictions of @{handle} starting with "{start}"...\nThis should take no more than 10 seconds!')
        
        with log_debug:
            try:                
                # start a wandb run (should never happen)
                if wandb.run is None:
                    run_name = handle_widget.value.strip()
                    %env WANDB_NAME=$run_name
                    wandb.init()

                wandb_url = wandb.run.get_url()
                
                # prepare input
                encoded_prompt = tokenizer.encode(start, add_special_tokens=False, return_tensors="pt")
                encoded_prompt = encoded_prompt.to(trainer.model.device)

                # prediction
                output_sequences = trainer.model.generate(
                    input_ids=encoded_prompt,
                    max_length=150,
                    min_length=20,
                    temperature=1.,
                    top_p=0.95,
                    do_sample=True,
                    num_return_sequences=20
                    )
                stop_token = '\n'
                generated_sequences = []

                # decode prediction
                for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                    generated_sequence = generated_sequence.tolist()
                    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                    text = text[: text.find(stop_token)]
                    generated_sequence = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                    generated_sequences.append(clean_prediction(generated_sequence))
                
                for i, g in enumerate(generated_sequences):
                    predictions.append([start, ' '.join([start, g])])
                
                # log predictions
                wandb.log({'examples': wandb.Table(data=predictions, columns=['Input', 'Prediction'])})

                # make html table
                tweet_data = [[tweet_html(g), ' '.join([start, g])] for g in generated_sequences]
                tweet_table = HTML(html_table(tweet_data))

                run_predictions.button_style = 'success'
                success_try = True
                
            except Exception as e:
                print('\nAn error occured...\n')
                print(e)
                run_predictions.button_style = 'danger'

        if success_try:
            with log_predictions:
                log_predictions.clear_output(wait=True)

                # display wandb run
                link = f'<a href="{wandb_url}" rel="noopener" target="_blank">{wandb_url}</a>'
                print_html(f'\n🚀 View all results under the "Media" panel at {link}\n')
                print_html('\n<b>Click on your favorite tweet or try new predictions with other sentences (or the same one)!</b>\n\n')
                
                # somehow display works one way with Jupyter and one way with colab
                if not IN_COLAB:
                    display(tweet_table)
                    print_html('\n<b>Click on your favorite tweet or try new predictions with other sentences (or the same one)!</b>\n\n')
            if IN_COLAB:
                display(tweet_table)
                print_html('\n<b>Click on your favorite tweet or try new predictions with other sentences (or the same one)!</b>\n\n')
        else:
            display(log_debug)
        
        run_predictions.disabled = False
        start_widget.disabled = False
                
    start_widget = widgets.Text(value='My dream is',
                                placeholder='Start a sentence')

    run_predictions = widgets.Button(
        description='Run predictions',
        button_style='primary',
        disabled=True)
    def on_run_predictions_clicked(b):
        predict()
    run_predictions.on_click(on_run_predictions_clicked)

    log_predictions = widgets.Output()
    with log_predictions:
        print_html('\nWaiting for Step 2 to complete...')

    clear_output(wait=True)
    print_html("🎉 Environment set-up correctly! You're ready to move to Step 1!")
