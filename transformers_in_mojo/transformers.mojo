fn main():
    # read it in to inspect it
    var f: FileHandle
    var text: String
    try:
        f = open('input.txt', 'rb')
        text = f.read()
    except:
        print("Failed")
        return

    try:
        f.close()
    except:
        return
    # print("length of dataset in characters: ", len(text))
    # print(text[:1000])

    # here are all the unique characters that occur in this text
    var chars = List[String]()
    var view = String()
    for i in range(len(text)):
        var char = String(text[i])
        if char not in chars:
            chars.append(char)
            view += char

    var vocab_size: Int = len(chars)
    # print(view)
    # print(vocab_size)

    var stoi = Dict[String,Int]()
    var itos = Dict[Int, String]()
    for i in range(len(chars)):
        stoi[chars[i]] = i
        itos[i] = chars[i]

    fn encode(s: String) -> List[Int]:
        var output = List[Int]()
        for i in range(len(s)):
            var c = Optional[Int]()
            c = stoi.get(s[i])
            if c:
                output.append(c.take())
            print(output[i])
        return output
    fn decode(list: List[Int]) -> String:
        var output = String()
        for i in range(len(list)):
            var c = Optional[String]()
            c = itos.get(list[i])
            if c:
                output += c.take()
        print(output)
        return output

    print(len(encode("hii there")))
    print(len(decode(encode("hii there"))))