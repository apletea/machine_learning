call plug#begin('~/.vim/plugged')
Plug 'autozimu/LanguageClient-neovim', {
        \ 'branch': 'next',
        \ 'do': 'bash install.sh',
        \ }
    Plug 'junegunn/fzf'
    Plug 'Shougo/deoplete.nvim', { 'do': ':UpdateRemotePlugins' }
    Plug 'vim-syntastic/syntastic'
    Plug 'w0rp/ale'
    Plug 'Shougo/vinarise.vim'
    Plug 'airblade/vim-gitgutter'
    Plug 'godlygeek/tabular'
    Plug 'haya14busa/incsearch.vim'
    Plug 'itchyny/lightline.vim'
    Plug 'jiangmiao/auto-pairs'
    Plug 'junegunn/vim-easy-align'
    Plug 'justinmk/vim-sneak'
    Plug 'lifepillar/vim-cheat40'
    Plug 'mhinz/vim-startify'
    Plug 'morhetz/gruvbox'
    Plug 'ntpeters/vim-better-whitespace'
    Plug 'scrooloose/nerdcommenter'
    Plug 'tpope/vim-fugitive'
    Plug 'tpope/vim-surround'
    Plug 'tpope/vim-unimpaired'
    Plug 'wellle/targets.vim'
    Plug 'scrooloose/nerdtree'
    Plug 'Xuyuanp/nerdtree-git-plugin'
    Plug 'jistr/vim-nerdtree-tabs'
    Plug 'SirVer/ultisnips'
    Plug 'honza/vim-snippets'
    Plug 'Shougo/echodoc.vim'
    Plug 'lervag/vimtex', { 'for': 'tex' }
    Plug 'octol/vim-cpp-enhanced-highlight', { 'for': ['cpp', 'c'] }
    Plug 'llvm-mirror/llvm', { 'rtp': 'utils/vim', 'for': 'llvm' }
    Plug 'Shirk/vim-gas'
    Plug 'junegunn/goyo.vim', { 'for': 'markdown' }
    Plug 'ryanoasis/vim-devicons'
    Plug 'vim-scripts/AutoComplPop'
    Plug 'Rip-Rip/clang_complete'
    Plug 'davidhalter/jedi-vim'
    Plug 'romainl/Apprentice'
call plug#end()
syntax on
filetype plugin indent on

set number
set tabstop=4 shiftwidth=4 expandtab
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*

let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0

set hidden
colorscheme apprentice

let g:LanguageClient_serverCommands = {
            \ "cpp": ["/home/apletea/clion-2018.2.4/bin/clang/linux/clangd"],
            \ "python": ["pyls"],
            \ }

nnoremap <silent> K :call LanguageClient#textDocument_hover()<CR>
nnoremap <silent> gd :call LanguageClient#textDocument_definition()<CR>
nnoremap <F5> :call LanguageClient_contextMenu()<CR>
nnoremap <silent> <F2> :call LanguageClient#textDocument_rename()<CR>
let g:deoplete_enable_on_startup = 1

"let g:acp_behavior={
			"\ 'c': 		[],
			"\ 'cpp': 	[],
			"\ 'python':  []}
"call add(g:acp_behavior.c, {
			"\ 	'command': 		"\<C-x>\<C-u>",
			"\ 	'completefunc': "ClangComplete",
			"\ 	'meets': 		"acp#meetsForKeyword",
			"\ 	'repeat': 		0,
			"\})
"call add(g:acp_behavior.cpp, {
			"\ 	'command': 		"\<C-x>\<C-u>",
			"\ 	'completefunc': "ClangComplete",
			"\ 	'meets': 		"acp#meetsForKeyword",
			"\ 	'repeat': 		0,
			"\})
