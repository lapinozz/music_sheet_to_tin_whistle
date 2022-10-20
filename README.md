# music_sheet_to_tin_whistle

Turns music sheets to tin whistle sheet:

<div style="display:flex">
<img src="/score_0.png" height="700" align="top">
<img src="/out/staffs_all.png" height="700">
</div>

It (partially) identifies the following features in the images

- The key
- The Signatures
- The notes
- The beams
- The flags
- Dots next to the notes
- The rest

It also annotates the images:
<img src="/debug/staff_1/staffImgAnnotated.png">


Usage:
 `python ./main.py ./path/to/sheet.png`

The program can also play the song using basic tones if you add `playTones` to the console line arguments or using recordings if you add `playRecordings`
