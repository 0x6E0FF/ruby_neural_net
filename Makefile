handwritten.exe: handwritten.c network.c
	gcc -Wall -g handwritten.c network.c -o handwritten.exe

network.c:
	ruby gen_c_table.rb trained_ruby_net > network.c

.PHONY: clean
clean:
	rm handwritten.exe network.c
